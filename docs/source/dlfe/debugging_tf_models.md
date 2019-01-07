# Debugging the TF Models

Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, 
you are, by definition, not smart enough to debug it. — BRIAN W. KERNIGHAN

## Eager Execution
- TODO

## Non Eager Execution

Debugging in general can be a tedious and challenging task. Nevertheless, you must be comfortable going through the 
written code and identifying problems. Normally there are many guides, and the process of debugging is often well documented for many languages and frameworks.

When it comes to TensorFlow, however, some new challenges arise because of the way it works.

As the official documentation states:

A TensorFlow Core program consists of two discrete sections:

- Building the computational graph (a tf.Graph).
- Running the computational graph (using a tf.Session).

~[](../../images/tf_graph.gif)

### Common Pradigms

- **Enable TF Logger**
```python
import tensorflow as tf
# Only log errors (to prevent unnecessary cluttering of the console)
tf.logging.set_verbosity(tf.logging.ERROR)
```
- [**TF Print**](https://www.tensorflow.org/api_docs/python/tf/Print)
```python
import tensorflow as tf
a = tf.constant([[1,1],[2,2]])
b = tf.constant([[1,1],[2,2]])
c = tf.constant([[1,1],[2,2]])
add = tf.add(a, b)
print_node = tf.Print(add, [a,b,add], message="Add: ")
out = tf.multiply(print_node, c)
```
Print output will show up in stderr in the console. Keep this in mind when searching for your print outputs!

Another word of caution: if you use tf.Print in your input function, be sure to limit the amount of data you pass in, 
otherwise you might end up scrolling through a very long console window :)

- [**TF Assert**](https://www.tensorflow.org/api_docs/python/tf/debugging/Assert)
```python
tf.Assert(...)
```

- **Tensorboard**
    - Proper tensor names and name scopes (`with tf.name_scope(): ....`)
    - Add tf.summaries
    - Add a tf.summary.FileWriter to create log files
    - Start the tensorboard server from your terminal
    ```
    For example: tensorboard --logdir=./logs/ --port=6006 --host=127.0.0.1
    ```
    - Navigating to the tensorboard server (in this case http://127.0.0.1:6006) 

- **Use the Tensorboard debugger**  

To accomplish this, there are 3 things to add to our previous example:

- Import `from tensorflow.python import debug as tf_debug`
- Add your session with `tf_debug.TensorBoardDebugWrapsperSession`
```python
import tensorflow as tf
from tensorflow.python import debug as tf_debug
sess = tf.Session()
sess = tf_debug.TensorBoardDebugWrapperSession(
    sess, "localhost:8080"
)
```
- Add to your tensorboard server the debugger_port

Now you have the option to debug the whole visualized model like with any other debugger, but with a beautiful map. 
You are able to select certain nodes and inspect them, control execution with the “step” and “continue” buttons, 
and visualize tensors and their values.
- [**Use the TensorFlow debugger**](https://www.tensorflow.org/guide/debugger)

### Simple Models
- Fetch and print values within Session.run

### Estimator Based Models
- Runtime Hooks

