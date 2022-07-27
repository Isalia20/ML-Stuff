import tensorflow as tf


@tf.autograph.experimental.do_not_convert
def triplet_loss(units, batch_size, margin):
    units = units
    batch_size = batch_size
    margin = margin

    def loss(y_true, y_pred):
        """
        Triplet loss with negative hard mining
        """
        anchor, positive = y_pred[:, :units], y_pred[:, units: units * 2]
        # new implementation
        scores = tf.linalg.matmul(anchor, tf.transpose(positive))
        positives = tf.linalg.tensor_diag_part(scores)
        negative_without_positive = scores - 2.0 * tf.eye(anchor.shape[0])
        closest_negative = tf.math.reduce_max(negative_without_positive, axis=1)
        negative_zero_on_duplicate = scores * (1.0 - tf.eye(batch_size))
        # We calculate mean negative that way as using reduce_mean would divide on 32 denominator
        mean_negative = tf.math.reduce_sum(negative_zero_on_duplicate, axis=1) / (anchor.shape[0] - 1)
        triplet_loss_1 = tf.math.maximum(0.0, margin - positives + closest_negative)
        triplet_loss_2 = tf.math.maximum(0.0, margin - positives + mean_negative)
        loss_score = tf.math.reduce_mean(triplet_loss_1 + triplet_loss_2)
        return loss_score

    return loss
