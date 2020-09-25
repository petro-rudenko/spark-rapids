package org.apache.spark.shuffle

import java.nio.ByteBuffer

import ai.rapids.cudf.MemoryBuffer
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.{MetaUtils, RapidsBufferId, ShuffleBufferId, ShuffleMetadata}
import org.openucx.jucx.UcxUtils

import org.apache.spark.shuffle.ucx.{Block, BlockId, MemoryBlock}
import org.apache.spark.storage.{ShuffleBlockBatchId, ShuffleBlockId}

case class RapidsBlockId(bufferId: RapidsBufferId) extends BlockId
case class RapidsMetaBlockId(shuffleBlockId: ShuffleBlockBatchId) extends BlockId
class RapidsMetaResponse(val bb: ByteBuffer)
    extends MemoryBlock(UcxUtils.getAddress(bb), bb.capacity())

class RapidsShuffleMetaBlock(
    shuffleBlockId: ShuffleBlockBatchId,
    tableMetas: Seq[TableMeta]) extends Block {

  val res = ShuffleMetadata.buildBlockMeta(tableMetas, 1024L * 1024L)

  override def getMemoryBlock: MemoryBlock = {
    MemoryBlock(UcxUtils.getAddress(res), res.capacity())
  }

  val rapidsMetaBlockId = RapidsMetaBlockId(shuffleBlockId)

  def getBlockId: RapidsMetaBlockId = rapidsMetaBlockId
}

class RapidsShuffleBlock(bufferId: ShuffleBufferId,
    buffer: MemoryBuffer, meta: TableMeta) extends Block {
  override def getMemoryBlock: MemoryBlock = {
    MemoryBlock(buffer.getAddress, buffer.getLength)
  }

  val rapidsBlockId = RapidsBlockId(bufferId)

  def getBlockId: RapidsBlockId = rapidsBlockId
}
