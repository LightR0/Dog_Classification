import os
import shutil

'''
删除训练txt中重复的delete_train_txt()，先print，手动删除。。。
'''
def delete_train_txt():
	file = open('E:/PyData/BaiDuDog/train.txt')
	lines = file.readlines()
	pic_id = []
	for line in lines:
		pic_id.append(line.split(' ')[0])
	print(len(pic_id))
	pic_id_1 = list(set(pic_id))
	print(len(pic_id_1))
	for i in pic_id_1:
		pic_id.remove(i)
	print(pic_id)
	file.close()
'''
训练数据分类train_devide_class()
'''
def train_devide_class():
	train_path = 'E:/PyData/BaiDuDog/train'
	save_path = 'E:/PyData/BaiDuDog/train_class'
	file = open('E:/PyData/BaiDuDog/train.txt')
	lines = file.readlines()
	for line in lines:
		pic_id = line.split(' ')[0] + '.jpg'
		label = line.split(' ')[1].split(' ')[0]
		if (os.path.exists(save_path + '/' + label) == False):
			os.makedirs(save_path + '/' + label)
		shutil.copy(train_path + '/' + pic_id,  save_path + '/' + label + '/')
'''
验证数据分类valid_devide_class()
'''
def valid_devide_class():
	train_path = 'E:/PyData/BaiDuDog/valid'
	save_path = 'E:/PyData/BaiDuDog/valid_class'
	file = open('E:/PyData/BaiDuDog/val.txt')
	lines = file.readlines()
	for line in lines:
		pic_id = line.split(' ')[0] + '.jpg'
		label = line.split(' ')[1].split(' ')[0]
		if (os.path.exists(save_path + '/' + label) == False):
			os.makedirs(save_path + '/' + label)
		shutil.copy(train_path + '/' + pic_id,  save_path + '/' + label + '/')
'''
构建验证集set_val()，从分类好的验证集里每个类别拿出18张构建本地验证集，然后把剩下的验证集复制到训练集中去，都进行训练
'''
def set_val():
	valid_path = 'E:/PyData/BaiDuDog/valid_class'
	save_path = 'E:/PyData/BaiDuDog/val_class'
	for classname in os.listdir(valid_path):
		tmp_path = valid_path + '/' + classname
		i = 0
		for pic in os.listdir(tmp_path):
			if i < 18 :
				if (os.path.exists(save_path+'/'+classname) == False):
					os.makedirs(save_path+'/'+classname)
				shutil.move(tmp_path+'/'+pic,save_path+'/'+classname+'/'+pic)
				i += 1
'''
训练集数据扩增DataAugmentation()
'''
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def DataAugmentation(path):
	datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
	j = 0
	for class_name in os.listdir(path):
		if j%20 == 0:
			print('leave:',100-j)
		j += 1
		tmp_path = path + '/' + class_name
		for pic in os.listdir(tmp_path):
			img = load_img(tmp_path + '/' + pic)
			x = img_to_array(img)
			x = x.reshape((1,) + x.shape)
			i = 0
			for batch in datagen.flow(x, batch_size=1,save_to_dir=tmp_path, save_prefix=pic.split('.')[0], save_format='jpg'):
				i += 1
				if i > 1:
					break
'''
测试集数据扩增testAugmentation()，测试集要增强2次，分开独立保存。每次要修改save_to_dir='E:/PyData/BaiDuDog/test_kz_2'分别为test_kz_1和test_kz_2
'''
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def testAugmentation(path):
	datagen = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
	for pic in os.listdir(path):
		img = load_img(path + '/' + pic)
		x = img_to_array(img)
		x = x.reshape((1,) + x.shape)
		i = 0
		for batch in datagen.flow(x, batch_size=1,save_to_dir='E:/PyData/BaiDuDog/test_kz_2', save_prefix=pic.split('.')[0], save_format='jpg'):
			i += 1
			if i > 0:
				break
