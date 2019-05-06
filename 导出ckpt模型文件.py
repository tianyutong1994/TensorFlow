import tensorflow as tf

saver = tf.train.Saver()

with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    #使用tf.train.write_graph导出GraphDef文件(结构图信息pb文件）
    tf.train.write_graph(sess.graph_def,"model/","nsfw-graph.pb",as_text=False)
    #tf.train.saver导出checkpoint文件（变量文件，实际产生四个文件）
    saver.save(sess,"model/nsfw_model.ckpt")
    
