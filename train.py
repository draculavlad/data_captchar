import tensorflow as tf
import numpy as np
import os, random, cv2

class Trainer():
    def __init__(self):

        self.train_data_path = './data_train'
        self.train_dict = {}
        self.train_imgs = []
        self.train_labels = []
        self.train_size = 0
        self.test_data_path = './data_test'
        self.test_dict = {}
        self.test_imgs = []
        self.test_labels = []
        self.test_size = 0

        self.log_path = './log'

        self.train_ptr = 0
        self.test_ptr = 0

        #image size
        self.img_w = 30
        self.img_h = 30

        # 最大迭代次数
        self.max_steps = 1000000
        self.train_size_dict = {}
        self.test_size_dict = {}

        self.charset_len = 37
        self.max_captcha = 1

        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

        # 指定cpu
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # 指定GPU
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        # config.gpu_options.allow_growth = True


        # 输入数据X占位符
        self.X = tf.placeholder(tf.float32, [None, self.img_h * self.img_w])
        # 输入数据Y占位符
        self.Y = tf.placeholder(tf.float32, [None, self.charset_len * self.max_captcha])
        # keepout占位符
        self.keep_prob = tf.placeholder(tf.float32)



    # 字符字典大小:0-9 a-z A-Z _(验证码如果小于4，用_补齐) 一共63个字符
    def get_imgs(self, train_path, test_path):
        for dir_name in os.listdir(train_path):
            dir_path = train_path + '/' + dir_name
            dir_imgs = os.listdir(dir_path)
            self.train_size += len(dir_imgs)
            for dir_img in dir_imgs:
                dir_img_path = dir_path + '/' + dir_img
                self.train_imgs.append(dir_img_path)
                self.train_dict[dir_img_path] = dir_name
        random.shuffle(self.train_imgs)
        self.train_labels = []
        for train_img in self.train_imgs:
            self.train_labels.append(self.train_dict[train_img])

        for dir_name in os.listdir(test_path):
            dir_path = test_path + '/' + dir_name
            dir_imgs = os.listdir(dir_path)
            self.test_size = self.test_size + len(dir_imgs)
            for dir_img in dir_imgs:
                dir_img_path = dir_path + '/' + dir_img
                self.test_imgs.append(dir_img_path)
                self.test_dict[dir_img_path] = dir_name
        random.shuffle(self.test_imgs)
        self.test_labels = []
        for test_img in self.test_imgs:
            self.test_labels.append(self.test_dict[test_img])


    def get_next_batch(self, train_flag=True, batch_size=100):
        """
        获得batch_size大小的数据集
        Parameters:
            batch_size:batch_size大小
            train_flag:是否从训练集获取数据
        Returns:
            batch_x:大小为batch_size的数据x
            batch_y:大小为batch_size的数据y
        """
        # 从训练集获取数据
        if train_flag == True:
            if (batch_size + self.train_ptr) < self.train_size:
                trains = self.train_imgs[self.train_ptr:(self.train_ptr + batch_size)]
                labels = self.train_labels[self.train_ptr:(self.train_ptr + batch_size)]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                trains = self.train_imgs[self.train_ptr:] + self.train_imgs[:new_ptr]
                labels = self.train_labels[self.train_ptr:] + self.train_labels[:new_ptr]
                train_ptr = new_ptr

            batch_x = np.zeros([batch_size, self.img_h * self.img_w])
            batch_y = np.zeros([batch_size, self.max_captcha * self.charset_len])
            try:
                for index, train in enumerate(trains):
                    img = np.mean(cv2.imread(train), -1)
                    # 将多维降维1维
                    batch_x[index, :] = img.flatten() / 255
                for index, label in enumerate(labels):
                    batch_y[index, :] = self.text2vec(label)
            except Exception as ex:
                raise ex

        # 从测试集获取数据
        else:
            if (batch_size + self.test_ptr) < self.test_size:
                tests = self.test_imgs[self.test_ptr:(self.test_ptr + batch_size)]
                labels = self.test_labels[self.test_ptr:(self.test_ptr + batch_size)]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                tests = self.test_imgs[self.test_ptr:] + self.test_imgs[:new_ptr]
                labels = self.test_labels[self.test_ptr:] + self.test_labels[:new_ptr]
                self.test_ptr = new_ptr

            batch_x = np.zeros([batch_size, self.img_h * self.img_w])
            batch_y = np.zeros([batch_size, self.max_captcha * self.charset_len])

            try:
                for index, test in enumerate(tests):
                    img = np.mean(cv2.imread(test), -1)
                    # 将多维降维1维
                    batch_x[index, :] = img.flatten() / 255
                for index, label in enumerate(labels):
                    batch_y[index, :] = self.text2vec(label)
            except Exception as ex:
                raise ex

        return batch_x, batch_y

    def text2vec(self, text):
        """
        文本转向量
        Parameters:
            text:文本
        Returns:
            vector:向量
        """
        if len(text) is not 1:
            raise ValueError('目标为单个字符')

        vector = np.zeros(1 * self.charset_len)

        def char2pos(c):
            if c == '_':
                k = 62
                return k
            k = ord(c) - 48
            if k > 9:
                k = ord(c) - 55
                if k > 35:
                    k = ord(c) - 61
                    if k > 61:
                        raise ValueError('No Map')
            return k

        for i, c in enumerate(text):
            idx = i * self.charset_len + char2pos(c)
            vector[idx] = 1
        print(vector)
        return vector


    def vec2text(self, vec):
        """
        向量转文本
        Parameters:
            vec:向量
        Returns:
            文本
        """
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_at_pos = i  # c/63
            char_idx = c % self.charset_len
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        print(text)
        return "".join(text)

    def crack_captcha_cnn(self, w_alpha=0.01, b_alpha=0.1):
        """
        定义CNN
        Parameters:
        	w_alpha:权重系数
        	b_alpha:偏置系数
        Returns:
        	out:CNN输出
        """
        x = tf.reshape(self.X, shape=[-1, self.img_h, self.img_w, 1])
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w_d = tf.Variable(w_alpha * tf.random_normal([4 * 4 * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)
        w_out = tf.Variable(w_alpha * tf.random_normal([1024, self.max_captcha * self.charset_len]))
        b_out = tf.Variable(b_alpha * tf.random_normal([self.max_captcha * self.charset_len]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out


    def train_crack_captcha_cnn(self):
        """
        训练函数
        """
        output = self.crack_captcha_cnn()

        # 创建损失函数
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.Y))
        diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.Y)
        loss = tf.reduce_mean(diff)
        tf.summary.scalar('loss', loss)

        # 使用AdamOptimizer优化器训练模型，最小化交叉熵损失
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # 计算准确率
        y = tf.reshape(output, [-1, self.max_captcha, self.charset_len])
        y_ = tf.reshape(self.Y, [-1, self.max_captcha, self.charset_len])
        correct_pred = tf.equal(tf.argmax(y, 2), tf.argmax(y_, 2))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            # 写到指定的磁盘路径中
            train_writer = tf.summary.FileWriter(self.log_path + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(self.log_path + '/test')
            sess.run(tf.global_variables_initializer())

            # 遍历self.max_steps次
            for i in range(self.max_steps):
                # 迭代500次，打乱一下数据集
                if i % 499 == 0:
                    self.get_imgs(self.train_data_path, self.test_data_path)
                # 每10次，使用测试集，测试一下准确率
                if i % 10 == 0:
                    batch_x_test, batch_y_test = self.get_next_batch(False, 100)
                    summary, acc = sess.run([merged, accuracy],
                                            feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1})
                    print('迭代第%d次 accuracy:%f' % (i + 1, acc))
                    test_writer.add_summary(summary, i)

                    # 如果准确率大于85%，则保存模型并退出。
                    if acc > 0.95:
                        train_writer.close()
                        test_writer.close()
                        saver.save(sess, "./model/crack_capcha.model", global_step=i)
                        break
                # 一直训练
                else:
                    batch_x, batch_y = self.get_next_batch(True, 100)
                    loss_value, _ = sess.run([loss, optimizer],
                                             feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1})
                    print('迭代第%d次 loss:%f' % (i + 1, loss_value))
                    curve = sess.run(merged, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1})
                    train_writer.add_summary(curve, i)

            train_writer.close()
            test_writer.close()
            saver.save(sess, "./model/crack_capcha.model", global_step=self.max_steps)


if __name__ == '__main__':
	tr = Trainer()
	tr.train_crack_captcha_cnn()
