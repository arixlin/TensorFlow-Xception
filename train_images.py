from builtins import range
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import xception_preprocessing
from xception import xception, xception_arg_scope
import os
import time
import model_data
slim = tf.contrib.slim

dataset_dir = './data'
log_dir = './log_face'
image_size = 299
IMG_W = image_size
IMG_H = image_size
num_classes = 16
CAPACITY = 2000
batch_size = 10
num_epochs = 64
rotate = False

VALRATIO = 0

initial_learning_rate = 0.1
learning_rate_decay_factor = 0.96
num_epochs_before_decay = 2


def run():
     if not os.path.exists(log_dir):
         os.mkdir(log_dir)

     #Training process

     with tf.Graph().as_default() as graph:
         tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

         train, train_label, val, val_label = model_data.get_path_files(dataset_dir, VALRATIO)
         train_batch, train_label_batch = model_data.get_batch(train,
                                                               train_label,
                                                               IMG_W,
                                                               IMG_H,
                                                               batch_size,
                                                               CAPACITY,
                                                               num_classes,
                                                               label_sytle='sparse',
                                                               rotate=rotate
                                                               )

         num_batches_per_epoch = len(train) // batch_size
         num_steps_per_epoch = num_batches_per_epoch
         decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

         with slim.arg_scope(xception_arg_scope()):
             logits, end_points = xception(train_batch, 16, is_training = True)

         one_hot_labels = slim.one_hot_encoding(train_label_batch, 16)

         loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
         total_loss = tf.losses.get_total_loss()

         global_step = get_or_create_global_step()


         print (num_steps_per_epoch * num_epochs)
         lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                         global_step=global_step,
                                         # decay_steps=decay_steps,
                                         decay_steps = 1000,
                                         decay_rate=learning_rate_decay_factor,
                                         staircase=True)

         optimizer = tf.train.AdamOptimizer(learning_rate = lr)

         train_op = slim.learning.create_train_op(total_loss, optimizer)

         # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
         predictions = tf.argmax(end_points['Predictions'], 1)
         probabilities = end_points['Predictions']
         accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, train_label_batch)
         metrics_op = tf.group(accuracy_update, probabilities)

         # Now finally create all the summaries you need to monitor and group them into one summary op.
         tf.summary.scalar('losses/Total_Loss', total_loss)
         tf.summary.scalar('accuracy', accuracy)
         tf.summary.scalar('learning_rate', lr)
         my_summary_op = tf.summary.merge_all()

         # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
         def train_step(sess, train_op, global_step):
             '''
             Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
             '''
             # Check the time for each sess run
             start_time = time.time()
             total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
             time_elapsed = time.time() - start_time

             # Run the logging to print some results
             logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

             return total_loss, global_step_count

         # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
         sv = tf.train.Supervisor(logdir=log_dir, summary_op=None)

         # Run the managed session
         with sv.managed_session() as sess:
             # for step in range(num_steps_per_epoch * num_epochs):
             for step in range(100000):
                 # At the start of every epoch, show the vital information:
                 if step % num_batches_per_epoch == 0:
                     logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                     learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                     logging.info('Current Learning Rate: %s', learning_rate_value)
                     logging.info('Current Streaming Accuracy: %s', accuracy_value)

                     # optionally, print your logits and predictions for a sanity check that things are going fine.
                     logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                         [logits, probabilities, predictions, train_label_batch])
                     print('logits: \n', logits_value[:5])
                     print('Probabilities: \n', probabilities_value[:5])
                     print('predictions: \n', predictions_value[:5])
                     print('Labels:\n:', labels_value[:5])

                 # Log the summaries every 10 step.
                 if step % 10 == 0:
                     loss, _ = train_step(sess, train_op, sv.global_step)
                     summaries = sess.run(my_summary_op)
                     sv.summary_computed(sess, summaries)

                 # If not, simply run the training step
                 else:
                     loss, _ = train_step(sess, train_op, sv.global_step)

             # We log the final training loss and accuracy
             logging.info('Final Loss: %s', loss)
             logging.info('Final Accuracy: %s', sess.run(accuracy))

             # Once all the training has been done, save the log files and checkpoint model
             logging.info('Finished training! Saving model to disk now.')


if __name__ == '__main__':
    run()