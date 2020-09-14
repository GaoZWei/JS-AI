// 直接用node运行的示例代码
const tf=require('@tensorflow/tfjs')
// const tf=require('@tensorflow/tfjs-node')  c++底层

const a=tf.tensor([1,2])
a.print()