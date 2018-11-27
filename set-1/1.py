import tensorflow as tf
# from tensorflow.python.tools import inspect_checkpoint as chkp
import os


W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inference(X):
    return tf.matmul(X,	W) + b


def loss(X,	Y):
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30],  [69, 25], [63, 28], [72, 36],
                  [79, 57], [75, 44], [27, 24], [89, 31], [
                      65, 52], [57, 23], [59, 60], [69, 48],
                  [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40]]
    blood_fat_content = [354, 190, 405, 263, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395,
                         434, 220, 374, 308, 220, 311, 181, 274, 303]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)

# def evaluate(sess, X, Y):
#     # print sess.run(inference([[63, 30]])) #244
#     # print sess.run(inference([[76, 57]])) #451
#     # print sess.run(inference([[80.,	25.]]))  # ~	303
#     # print sess.run(inference([[65.,	25.]]))  # ~	256

def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)




with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X,	Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()  # ???
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # ???

    saver = tf.train.Saver()

    initial_step = 0
    chpt = tf.train.latest_checkpoint(os.path.dirname(__file__))
    if chpt and chpt.model_checkpoint_path:
        saver.restore(sess, chpt.model_checkpoint_path)
        initial_step = int(chpt.chpt.model_checkpoint_path.rsplit('-', 1)[1])
        print('read')

    training_steps = 1000
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        if step % 10 == 0:
            print( "loss:	",	sess.run([total_loss]))
        if step % 10000 == 0:
            saver.save(sess, '../tf-exp1', global_step=step)

    # After Train
    print( sess.run(inference([[80.,	25.]])))  # ~	303
    print( sess.run(inference([[65.,	25.]])))  # ~	256
    coord.request_stop()
    coord.join(threads)

    saver.save(sess, '../tf-exp1', global_step=training_steps)

    sess.close()
