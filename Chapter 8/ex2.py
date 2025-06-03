import tensorflow as tf

tf.random.set_seed(1)
t = tf.random.uniform((6,))
print(t.numpy())
t_splits = tf.split(t,3) # tách tensor 6 làm 3 phần
[item.numpy() for item in t_splits]

A = tf.ones((3,))
B = tf.zeros((2,))
C = tf.concat([A,B],axis=0)
print(C.numpy())

A = tf.ones((3,))
B = tf.zeros((3,))
S = tf.stack([A,B],axis=1)
print(S.numpy())

a = [1.2,3.4,7.5,4.1,5.0,1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)

for item in ds:
  print(item)

ds_batch = ds.batch(3)
for i, elem in enumerate(ds_batch,100):
  print('batch {}:'.format(i),elem.numpy())

## Join 2 tensor
tf.random.set_seed(1)
t_x = tf.random.uniform([4,3],dtype=tf.float32)
t_y = tf.range(4)
print(t_x)
print(t_y)

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)
ds_joint = tf.data.Dataset.zip((ds_x,ds_y))
for item in ds_joint:
  print(' x:', item[0].numpy(),
        ' y:',item[1].numpy())

ds_trans = ds_joint.map(lambda x,y:(x*2-1.0,y))
for item in ds_trans:
  print(' x:', item[0].numpy(),
        ' y:',item[1].numpy())

## shuffle(), batch(), repeat()
tf.random.set_seed(1)
ds = ds_joint.shuffle(buffer_size=len(t_x))
for e in ds:
  print(' x:', e[0].numpy(),
        ' y:',e[1].numpy())

ds = ds_joint.batch(batch_size=3,drop_remainder=False)
batch_x,batch_y = next(iter(ds))
print('batch x:\n',batch_x.numpy())
print('batch y:\n',batch_y.numpy())

ds = ds_joint.batch(3).repeat(count=2)
for i,(batch_x,batch_y) in enumerate(ds):
  print(i,batch_x.numpy(),batch_y.numpy())

tf.random.set_seed(1)
ds = ds_joint.shuffle(4).batch(2).repeat(20)
for i,(batch_x,batch_y) in enumerate(ds):
  print(i,batch_x.numpy(),batch_y.numpy())