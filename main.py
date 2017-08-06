from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py
'''
训练集和验证集生成特征文件write_gap()
'''
def write_gap(MODEL, gap_name, image_size, lambda_func=None):
	width = image_size[0]
	height = image_size[1]
	input_tensor = Input((height, width, 3))
	x = input_tensor
	if lambda_func:
		x = Lambda(lambda_func)(x)
	base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
	model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
	gen = ImageDataGenerator()
	train_generator = gen.flow_from_directory("E:/PyData/BaiDuDog/train_class", image_size, shuffle=False, batch_size=32)
	val_generator = gen.flow_from_directory("E:/PyData/BaiDuDog/val_class", image_size, shuffle=False, batch_size=32)
	train = model.predict_generator(train_generator, train_generator.samples//32+1,verbose=1)
	valid = model.predict_generator(val_generator, val_generator.samples//32+1, verbose=1)
	with h5py.File(gap_name) as h:
		h.create_dataset("train", data=train)
		h.create_dataset("valid", data=valid)
		h.create_dataset("y_train", data=train_generator.classes)
		h.create_dataset("y_valid", data=val_generator.classes)
	return train_generator.class_indices
  class_dict = write_gap(InceptionV3, "gap_t_v_InceptionV3.h5",(299, 299), inception_v3.preprocess_input)
  class_dict = write_gap(Xception, "gap_t_v_Xception.h5", (299, 299),xception.preprocess_input)
  '''
测试集生成特征文件test_write_gap()
'''
def test_write_gap(MODEL, gap_name, path, image_size, lambda_func=None):
	width = image_size[0]
	height = image_size[1]
	input_tensor = Input((height, width, 3))
	x = input_tensor
	if lambda_func:
		x = Lambda(lambda_func)(x)
	base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
	model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
	gen = ImageDataGenerator()
	test_generator = gen.flow_from_directory(path, image_size, shuffle=False, batch_size=32, class_mode=None)
	test = model.predict_generator(test_generator, test_generator.samples//32+1,verbose=1)
	with h5py.File(gap_name) as h:
		h.create_dataset("test", data=test)
 test_write_gap(InceptionV3, "gap_test_InceptionV3.h5","E:/PyData/BaiDuDog/test",(299, 299), inception_v3.preprocess_input)
 test_write_gap(Xception, "gap_test_Xception.h5","E:/PyData/BaiDuDog/test",(299, 299), xception.preprocess_input)
 test_write_gap(InceptionV3, "gap_test1_InceptionV3.h5","E:/PyData/BaiDuDog/test1",(299, 299), inception_v3.preprocess_input)
 test_write_gap(Xception, "gap_test1_Xception.h5","E:/PyData/BaiDuDog/test1",(299, 299), xception.preprocess_input)
 test_write_gap(InceptionV3, "gap_test2_InceptionV3.h5","E:/PyData/BaiDuDog/test2",(299, 299), inception_v3.preprocess_input)
 test_write_gap(Xception, "gap_test2_Xception.h5","E:/PyData/BaiDuDog/test2",(299, 299), xception.preprocess_input)
 '''
载入特征向量load_feature()
'''
import h5py
import numpy as np
import keras
from sklearn.utils import shuffle
np.random.seed(2017)

X_train = []
X_valid = []
for filename in ["gap_t_v_InceptionV3.h5", "gap_t_v_Xception.h5"]:
	with h5py.File(filename, 'r') as h:
		X_train.append(np.array(h['train']))
		y_train = np.array(h['y_train'])
		X_valid.append(np.array(h['valid']))
		y_valid = np.array(h['y_valid'])
X_train = np.concatenate(X_train, axis=1)
y_train = keras.utils.to_categorical(y_train, 100)
X_train, y_train = shuffle(X_train, y_train)
X_valid = np.concatenate(X_valid, axis=1)
y_valid = keras.utils.to_categorical(y_valid, 100)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test = []
for filename in ["gap_test_InceptionV3.h5", "gap_test_Xception.h5"]:
	with h5py.File(filename, 'r') as h:
		X_test.append(np.array(h['test']))
X_test = np.concatenate(X_test, axis=1)
X_test1 = []
for filename in ["gap_test1_InceptionV3.h5", "gap_test1_Xception.h5"]:
	with h5py.File(filename, 'r') as h:
		X_test1.append(np.array(h['test']))
X_test1 = np.concatenate(X_test1, axis=1)
X_test2 = []
for filename in ["gap_test2_InceptionV3.h5", "gap_test2_Xception.h5"]:
	with h5py.File(filename, 'r') as h:
		X_test2.append(np.array(h['test']))
X_test2 = np.concatenate(X_test2, axis=1)
print(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape,X_test.shape,X_test1.shape,X_test2.shape)
'''
构建模型
'''
from keras.models import *
from keras.layers import *
np.random.seed(2017)

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(100, activation='softmax')(x)
model = Model(input_tensor, x)
model.compile(optimizer='Adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
'''
训练
'''
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid,y_valid))
'''
预测
'''
y_pred0 = model.predict(X_test, verbose=1)
'''
预测
'''
y_pred1 = model.predict(X_test1, verbose=1)
'''
预测
'''
y_pred2 = model.predict(X_test2, verbose=1)
y_pred = (y_pred0+y_pred1+y_pred2)/3
'''
取每行概率最大索引
'''
label_index = []
for i in y_pred:
	label_index.append(list(i).index(max(list(i))))
'''
取图片id
'''
test_ID = []
gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("E:/PyData/BaiDuDog/test", (224, 224), shuffle=False, batch_size=16, class_mode=None)
for i,fname in enumerate(test_generator.filenames):
	test_ID.append(fname[fname.rfind('\\')+1:fname.rfind('.')])
'''
写入txt
'''
file = open('E:/PyData/BaiDuDog/sample_sub.txt','w')
for i in range(10593):
	file.writelines(list(class_dict.keys())[list(class_dict.values()).index(label_index[i])])
	file.writelines('\t')
	file.writelines(test_ID[i])
	file.writelines('\n')
file.close()
