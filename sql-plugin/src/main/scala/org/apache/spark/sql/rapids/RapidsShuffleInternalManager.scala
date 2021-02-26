/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.rapids

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Success

import ai.rapids.cudf.{DeviceMemoryBuffer, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.shuffle.{BounceBufferManager, RapidsShuffleTransport}

import org.apache.spark.internal.Logging
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.rpc.RpcEnv
import org.apache.spark.scheduler.MapStatus
import org.apache.spark.shuffle._
import org.apache.spark.shuffle.sort.SortShuffleManager
import org.apache.spark.shuffle.ucx.rpc.UcxRpcMessages.{ExecutorAdded, IntroduceAllExecutors}
import org.apache.spark.shuffle.ucx.rpc.{UcxDriverRpcEndpoint, UcxExecutorRpcEndpoint}
import org.apache.spark.shuffle.ucx.utils.SerializableDirectBuffer
import org.apache.spark.shuffle.ucx.{Block, MemoryBlock, ShuffleTransport, UcxShuffleTransport}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage._
import org.apache.spark.util.RpcUtils
import org.apache.spark.{SecurityManager, ShuffleDependency, SparkConf, SparkEnv, TaskContext}

class GpuShuffleHandle[K, V](
    val wrapped: ShuffleHandle,
    override val dependency: GpuShuffleDependency[K, V, V])
  extends BaseShuffleHandle(wrapped.shuffleId, dependency) {

  override def toString: String = s"GPU SHUFFLE HANDLE $shuffleId"
}

class GpuShuffleBlockResolver(
    private val wrapped: ShuffleBlockResolver,
    catalog: ShuffleBufferCatalog)
  extends ShuffleBlockResolver with Logging {
  override def getBlockData(blockId: BlockId, dirs: Option[Array[String]]): ManagedBuffer = {
    val hasActiveShuffle: Boolean = blockId match {
      case sbbid: ShuffleBlockBatchId =>
        catalog.hasActiveShuffle(sbbid.shuffleId)
      case sbid: ShuffleBlockId =>
        catalog.hasActiveShuffle(sbid.shuffleId)
      case _ => throw new IllegalArgumentException(s"${blockId.getClass} $blockId "
          + "is not currently supported")
    }
    if (hasActiveShuffle) {
      throw new IllegalStateException(s"The block $blockId is being managed by the catalog")
    }
    wrapped.getBlockData(blockId)
  }

  override def stop(): Unit = wrapped.stop()
}


object RapidsShuffleInternalManagerBase extends Logging {
  def unwrapHandle(handle: ShuffleHandle): ShuffleHandle = handle match {
    case gh: GpuShuffleHandle[_, _] => gh.wrapped
    case other => other
  }
}

class RapidsCachingWriter[K, V](
    blockManager: BlockManager,
    // Never keep a reference to the ShuffleHandle in the cache as it being GCed triggers
    // the data being released
    handle: GpuShuffleHandle[K, V],
    mapId: Long,
    metricsReporter: ShuffleWriteMetricsReporter,
    catalog: ShuffleBufferCatalog,
    shuffleStorage: RapidsDeviceMemoryStore,
    transport: ShuffleTransport,
    metrics: Map[String, SQLMetric]) extends ShuffleWriter[K, V] with Logging {

  private val numParts = handle.dependency.partitioner.numPartitions
  private val sizes = new Array[Long](numParts)
  private val writtenBufferIds = new ArrayBuffer[ShuffleBufferId](numParts)
  private val uncompressedMetric: SQLMetric = metrics("dataSize")

  private val blocksWritten = mutable.HashSet[ShuffleBlockId]()

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    val nvtxRange = new NvtxRange("RapidsCachingWriter.write", NvtxColor.CYAN)
    try {
      var bytesWritten: Long = 0L
      var recordsWritten: Long = 0L
      records.foreach { p =>
        val partId = p._1.asInstanceOf[Int]
        val batch = p._2.asInstanceOf[ColumnarBatch]
        logDebug(s"Caching shuffle_id=${handle.shuffleId} map_id=$mapId, partId=$partId, "
            + s"batch=[num_cols=${batch.numCols()}, num_rows=${batch.numRows()}]")
        recordsWritten = recordsWritten + batch.numRows()
        var partSize: Long = 0
        val blockId = ShuffleBlockId(handle.shuffleId, mapId, partId)
        val bufferId = catalog.nextShuffleBufferId(blockId)
        if (batch.numRows > 0 && batch.numCols > 0) {
          // Add the table to the shuffle store
          batch.column(0) match {
            case c: GpuColumnVectorFromBuffer =>
              val buffer = c.getBuffer
              buffer.incRefCount()
              partSize = buffer.getLength
              uncompressedMetric += partSize
              shuffleStorage.addTable(
                bufferId,
                GpuColumnVector.from(batch),
                buffer,
                SpillPriorities.OUTPUT_FOR_SHUFFLE_INITIAL_PRIORITY)
            case c: GpuCompressedColumnVector =>
              val buffer = c.getBuffer
              buffer.incRefCount()
              partSize = buffer.getLength
              val tableMeta = c.getTableMeta
              // update the table metadata for the buffer ID generated above
              tableMeta.bufferMeta.mutateId(bufferId.tableId)
              uncompressedMetric += tableMeta.bufferMeta().uncompressedSize()
              shuffleStorage.addBuffer(
                bufferId,
                buffer,
                tableMeta,
                SpillPriorities.OUTPUT_FOR_SHUFFLE_INITIAL_PRIORITY)
            case c => throw new IllegalStateException(s"Unexpected column type: ${c.getClass}")
          }
          blocksWritten += blockId
          bytesWritten += partSize
          sizes(partId) += partSize
        } else {
          // no device data, tracking only metadata
          val tableMeta = MetaUtils.buildDegenerateTableMeta(batch)
          catalog.registerNewBuffer(new DegenerateRapidsBuffer(bufferId, tableMeta))

          // The size of the data is really only used to tell if the data should be shuffled or not
          // a 0 indicates that we should not shuffle anything.  This is here for the special case
          // where we have no columns, because of predicate push down, but we have a row count as
          // metadata.  We still want to shuffle it. The 100 is an arbitrary number and can be
          // any non-zero number that is not too large.
          if (batch.numRows > 0) {
            sizes(partId) += 100
          }
        }
        writtenBufferIds.append(bufferId)
      }
      metricsReporter.incBytesWritten(bytesWritten)
      metricsReporter.incRecordsWritten(recordsWritten)
    } finally {
      nvtxRange.close()
    }
  }

  /**
   * Used to remove shuffle buffers when the writing task detects an error, calling `stop(false)`
   */
  private def cleanStorage(): Unit = {
    writtenBufferIds.foreach(catalog.removeBuffer)
  }

  override def stop(success: Boolean): Option[MapStatus] = {
    val nvtxRange = new NvtxRange("RapidsCachingWriter.close", NvtxColor.CYAN)
    try {
      if (!success) {
        cleanStorage()
        None
      } else {
        // upon seeing this port, the other side will try to connect to the port
        // in order to establish an UCX endpoint (on demand), if the topology has "rapids" in it.
        blocksWritten.foreach { bId =>
          val bufferIds: Array[ShuffleBufferId] = catalog.blockIdToBuffersIds(bId)
          bufferIds.foreach { bId =>
            val buffer = catalog.acquireBuffer(bId)
            val actualBuffer = buffer.getMemoryBuffer
            val transportBlock =
              new RapidsShuffleBlock(bId, actualBuffer.getAddress,
                actualBuffer.getLength, buffer.meta)
            actualBuffer.close()
            transport.register(transportBlock.rapidsBlockId, transportBlock)
            buffer.close()
          }

          val metas = catalog.blockIdToMetas(bId)

          val sbId = bId match {
            case sbbid: ShuffleBlockBatchId => sbbid
            case sbid: ShuffleBlockId =>
                ShuffleBlockBatchId(sbid.shuffleId, sbid.mapId, sbid.reduceId, sbid.reduceId)
            case _ =>
              throw new IllegalArgumentException(
                s"${bId.getClass} $bId is not currently supported")
          }
          val rapidsMeta = new RapidsShuffleMetaBlock(sbId, metas)
          transport.register(rapidsMeta.getBlockId, rapidsMeta)
        }
        val shuffleServerId = if (transport != null) {
          val originalShuffleServerId = blockManager.blockManagerId
          BlockManagerId(
            originalShuffleServerId.executorId,
            originalShuffleServerId.host,
            originalShuffleServerId.port,
            Some(s"${RapidsShuffleTransport.BLOCK_MANAGER_ID_TOPO_PREFIX}=spark-ucx"))
        } else {
          blockManager.shuffleServerId
        }
        logInfo(s"Done caching shuffle success=$success, server_id=$shuffleServerId, "
            + s"map_id=$mapId, sizes=${sizes.mkString(",")}")
        Some(MapStatus(shuffleServerId, sizes, mapId))
      }
    } finally {
      nvtxRange.close()
    }
  }
}

/**
 * A shuffle manager optimized for the RAPIDS Plugin For Apache Spark.
 * @note This is an internal class to obtain access to the private
 *       `ShuffleManager` and `SortShuffleManager` classes. When configuring
 *       Apache Spark to use the RAPIDS shuffle manager,
 */
abstract class RapidsShuffleInternalManagerBase(conf: SparkConf, isDriver: Boolean)
    extends ShuffleManager with Logging {

  private val rapidsConf = new RapidsConf(conf)

  // set the shim override if specified since the shuffle manager loads early
  if (rapidsConf.shimsProviderOverride.isDefined) {
    ShimLoader.setSparkShimProviderClass(rapidsConf.shimsProviderOverride.get)
  }

  protected val wrapped = new SortShuffleManager(conf)

  GpuShuffleEnv.setRapidsShuffleManagerInitialized(true, this.getClass.getCanonicalName)

  private [this] val transportEnabledMessage = if (!rapidsConf.shuffleTransportEnabled) {
    "Transport disabled (local cached blocks only)."
  } else {
    s"Transport enabled (remote fetches will use ${rapidsConf.shuffleTransportClassName})."
  }

  logWarning(s"Rapids Shuffle Plugin enabled. ${transportEnabledMessage}")

  //Many of these values like blockManager are not initialized when the constructor is called,
  // so they all need to be lazy values that are executed when things are first called

  // NOTE: this can be null in the driver side.
  private lazy val env = SparkEnv.get
  private lazy val blockManager = env.blockManager
  private lazy val shouldFallThroughOnEverything = {
    val fallThroughReasons = new ListBuffer[String]()
    if (!GpuShuffleEnv.isRapidsShuffleEnabled) {
      fallThroughReasons += "external shuffle is enabled"
    }
    if (fallThroughReasons.nonEmpty) {
      logWarning(s"Rapids Shuffle Plugin is falling back to SortShuffleManager " +
          s"because: ${fallThroughReasons.mkString(", ")}")
    }
    fallThroughReasons.nonEmpty
  }

  private lazy val localBlockManagerId = blockManager.blockManagerId

  // Code that expects the shuffle catalog to be initialized gets it this way,
  // with error checking in case we are in a bad state.
  private def getCatalogOrThrow: ShuffleBufferCatalog =
    Option(GpuShuffleEnv.getCatalog).getOrElse(
      throw new IllegalStateException("The ShuffleBufferCatalog is not initialized but the " +
        "RapidsShuffleManager is configured"))

  private lazy val resolver = if (shouldFallThroughOnEverything) {
    wrapped.shuffleBlockResolver
  } else {
    new GpuShuffleBlockResolver(wrapped.shuffleBlockResolver, getCatalogOrThrow)
  }

  private[this] lazy val transport: Option[ShuffleTransport] = {
    if (rapidsConf.shuffleTransportEnabled && !isDriver) {
      val transport =
        RapidsShuffleTransport.makeTransport(blockManager.shuffleServerId, rapidsConf)
      initUcxTransport(transport.asInstanceOf[UcxShuffleTransport])
      RapidsBufferCatalog.setTransport(transport)
      Some(transport)
    } else {
      None
    }
  }

  private val deviceReceiveBuffMgr =
    new BounceBufferManager[DeviceMemoryBuffer](
      "device-receive",
      rapidsConf.shuffleUcxBounceBuffersSize,
      rapidsConf.shuffleUcxDeviceBounceBuffersCount,
      (size: Long) => DeviceMemoryBuffer.allocate(size))
  private val bounceBuferBlockId = RapidsMetaBlockId(ShuffleBlockId(-1, -1, -1).name)

  @volatile private var initialized: Boolean = false
  private val driverEndpointName = "ucx-shuffle-driver"

  // TODO: initialize through IO plugin at the process start
  private def initDriverRpc(): Unit = {
    if (!initialized) {
      val rpcEnv = SparkEnv.get.rpcEnv
      val driverEndpoint = new UcxDriverRpcEndpoint(rpcEnv)
      rpcEnv.setupEndpoint(driverEndpointName, driverEndpoint)
      initialized = true
    }
  }

  private def initUcxTransport(ucxTransport: UcxShuffleTransport): Unit = this.synchronized {
    if (!initialized) {
      val blockManager = SparkEnv.get.blockManager.blockManagerId
      val rpcEnv = RpcEnv.create("ucx-rpc-env", blockManager.host, blockManager.host,
        blockManager.port, conf, new SecurityManager(conf), 1, clientMode = false)
      logDebug("Initializing ucx transport")
      val address = ucxTransport.init()
      val executorEndpoint = new UcxExecutorRpcEndpoint(rpcEnv, ucxTransport)
      val endpoint = rpcEnv.setupEndpoint(
        s"ucx-shuffle-executor-${blockManager.executorId}",
        executorEndpoint)

      val driverEndpoint = RpcUtils.makeDriverRef(driverEndpointName, conf, rpcEnv)
      driverEndpoint.ask[IntroduceAllExecutors](ExecutorAdded(blockManager.executorId,
        endpoint, new SerializableDirectBuffer(address)))
        .andThen {
          case Success(msg) =>
            logInfo(s"Receive reply $msg")
            executorEndpoint.receive(msg)
        }
      ucxTransport.register(bounceBuferBlockId,
        new Block {
          override def getMemoryBlock: MemoryBlock = {
            val rootBoofer = deviceReceiveBuffMgr.getRootBuffer()
            MemoryBlock(rootBoofer.getAddress, rootBoofer.getLength, isHostMemory = false)
          }
        })
      initialized = true
    }
  }


  override def registerShuffle[K, V, C](shuffleId: Int,
                                        dependency: ShuffleDependency[K, V, C]): ShuffleHandle = {

    // Always register with the wrapped handler so we can write to it ourselves if needed
    initDriverRpc()
    val orig = wrapped.registerShuffle(shuffleId, dependency)
    if (!shouldFallThroughOnEverything && dependency.isInstanceOf[GpuShuffleDependency[K, V, C]]) {
      val handle = new GpuShuffleHandle(orig,
        dependency.asInstanceOf[GpuShuffleDependency[K, V, V]])
      handle
    } else {
      orig
    }
  }

  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metricsReporter: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] = {
    handle match {
      case gpu: GpuShuffleHandle[_, _] =>
        registerGpuShuffle(handle.shuffleId)
        new RapidsCachingWriter(
          env.blockManager,
          gpu.asInstanceOf[GpuShuffleHandle[K, V]],
          mapId,
          metricsReporter,
          getCatalogOrThrow,
          RapidsBufferCatalog.getDeviceStorage,
          transport.get,
          gpu.dependency.metrics)
      case other =>
        wrapped.getWriter(other, mapId, context, metricsReporter)
    }
  }

  def getReaderInternal[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    handle match {
      case gpu: GpuShuffleHandle[_, _] =>
        logInfo(s"Asking map output tracker for dependency ${gpu.dependency}, " +
          s"map output sizes for: ${gpu.shuffleId}, parts=$startPartition-$endPartition")
        if (gpu.dependency.keyOrdering.isDefined) {
          // very unlikely, but just in case
          throw new IllegalStateException("A key ordering was requested for a gpu shuffle "
              + s"dependency ${gpu.dependency.keyOrdering.get}, this is not supported.")
        }

        val nvtxRange = new NvtxRange("getMapSizesByExecId", NvtxColor.CYAN)
        val blocksByAddress = try {
          ShimLoader.getSparkShims.getMapSizesByExecutorId(gpu.shuffleId,
            startMapIndex, endMapIndex, startPartition, endPartition)
        } finally {
          nvtxRange.close()
        }

        new RapidsCachingReader(rapidsConf, localBlockManagerId,
          blocksByAddress,
          context,
          metrics,
          transport,
          getCatalogOrThrow,
          gpu.dependency.sparkTypes,
          deviceReceiveBuffMgr)
      case other => {
        val shuffleHandle = RapidsShuffleInternalManagerBase.unwrapHandle(other)
        wrapped.getReader(shuffleHandle, startPartition, endPartition, context, metrics)
      }
    }
  }

  def registerGpuShuffle(shuffleId: Int): Unit = {
    val catalog = GpuShuffleEnv.getCatalog
    if (catalog != null) {
      // Note that in local mode this can be called multiple times.
      logInfo(s"Registering shuffle $shuffleId")
      catalog.registerShuffle(shuffleId)
    }
  }

  def unregisterGpuShuffle(shuffleId: Int): Unit = {
    val catalog = GpuShuffleEnv.getCatalog
    if (catalog != null) {
      logInfo(s"Unregistering shuffle $shuffleId")
      catalog.unregisterShuffle(shuffleId)
    }
  }

  override def unregisterShuffle(shuffleId: Int): Boolean = {
    unregisterGpuShuffle(shuffleId)
    wrapped.unregisterShuffle(shuffleId)
  }

  override def shuffleBlockResolver: ShuffleBlockResolver = resolver

  override def stop(): Unit = {
    wrapped.stop()
    transport.foreach(t => t.unregister(bounceBuferBlockId))
    transport.foreach(_.close())
  }
}
