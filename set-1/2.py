import tensorflow as tf
# from tensorflow.python.tools import inspect_checkpoint as chkp
import os


W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


def combine_inputs(X):
    return tf.matmul(X,	W) + b


def inference(X):
    return tf.sigmoid(combine_inputs(X))


def loss(sess, X,	Y):
    Y_predicted = inference(X)
    # print("Y : ", sess.run(Y), " , logit : ", sess.run(Y_predicted))
    return tf.reduce_min(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y, name="logits"))


def read_csv(batch_size,	file_name,	record_defaults):
    filename_queue = tf.train.string_input_producer(
        [os.path.dirname(__file__) + "/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key,	value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = read_csv(100,	"./data/train.csv",	[[0.0], [0.0], [0],	[""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""],	[""]])
    is_first_class = tf.to_float(tf.equal(pclass,	[1]))
    is_second_class = tf.to_float(tf.equal(pclass,	[2]))
    is_third_class = tf.to_float(tf.equal(pclass,	[3]))
    gender = tf.to_float(tf.equal(sex,	["female"]))
    features = tf.transpose(tf.stack([is_first_class,	is_second_class,	is_third_class,	gender,	age]))
    survived = tf.reshape(survived,	[100,	1])
    return features,	survived


def evaluate(sess,	X,	Y):
    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print( sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted,	Y),	tf.float32))))


def train(total_loss):
    learning_rate = 0.02
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X,	Y = inputs()
    total_loss = loss(sess,X, Y)
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

    training_steps = 100000
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        if step % 100 == 0:
            print("loss:	",	sess.run([total_loss]))
            evaluate(sess, X, Y)

        # if step % 10000 == 0:
        #     saver.save(sess, '../tf-exp2', global_step=step)

    # After Train
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)

    saver.save(sess, '../tf-exp2', global_step=training_steps)

    sess.close()