from dynamic_sparse_image_warp import sparse_image_warp
import tensorflow as tf

def time_warp(mel_fbanks, W=80):
  """Time warping using dynamic sparse_image_warp
  instead of `tf.contrib.image.sparse_image_warp`.
  Args:
    mel_fbanks : `[1, tau, v, 1]` float `Tensor`
    W : time warp parameter 
  """
  
  tau, v = (tf.shape(mel_fbanks)[1], tf.shape(mel_fbanks)[2])
  rp = tf.random_uniform([], W, tau - W, tf.int32)
  cp = tf.expand_dims(v // 2)
  w = tf.random_uniform([], 0, W, tf.float32)
  
  src_ctp_loc = tf.stack((cp, tf.ones_like(cp) * rp), -1)
  src_ctp_loc = tf.to_float(tf.expand_dims(src_ctp_loc, 0))
  
  dest_ctp_loc = tf.stack((cp, tf.ones_like(cp) * (tf.to_float(rp) + w)), -1)
  dest_ctp_loc = tf.to_float(tf.expand_dims(dest_ctp_loc, 0))
  mel_fbanks = tf.transpoase(mel_fbanks, perm=[0, 2, 1, 3])
  warped_mel_fbanks, _ = sparse_image_warp(mel_fbanks,
                                           source_control_point_locations=src_ctp_loc,
                                           dest_control_point_locations=dest_ctp_loc,
                                           num_boundary_points=1)
  warped_mel_fbanks = tf.transpoase(warped_mel_fbanks, perm=[0, 2, 1, 3]
  return warped_mel_fbanks
