import logging
logging.getLogger('albumentations').setLevel(logging.WARNING) #禁止显示albumentations更新信息

#SimSwap里的
import cv2
import torch
import fractions                                #数学包,“分数、无穷精度、实数”
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms,utils
#SBI里的
import dlib
from imutils import face_utils
import albumentations as alb
alb.check_version = lambda: None #禁用版本检查，免得警告报错啥的
import logging
#本程序用到的
import os
import sys
from glob import glob #文件处理
from tqdm import tqdm #进度条
import random
import time #计时

#设置随机种子,可复现
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#设置CuDNN为确定性模式
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
#设置Python环境变量
os.environ['PYTHONHASHSEED']=str(seed)

##一些设置
image_size=(224,224) #图片大小

input_path='D:\\my\\dataset\\vggface2_crop_arcfacealign_224\\' #输入文件路径
output_path='.\\synthesis\\' #输出文件路径
already_folder_num=0 #已处理子文件夹总数
folder_num=10 #待处理子文件夹总数

##所使用模型导入
#选择设备
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

#文件路径
code_path=os.path.dirname(__file__) #程序路径,后面修改工作路径我用的绝对路径
work_path=os.getcwd() #当前工作路径
SimSwap_path=code_path+'\\SimSwap' #SimSwap的工作路径
SBI_path=code_path+'\\SelfBlendedImages' #SBI的工作路径

#SimSwap实例化模型,导入文件模块
os.chdir(SimSwap_path) #更新工作目录
sys.path.append(SimSwap_path) #更新模块导入目录
from models.models import create_model          #创建simswap模型的函数
from options.test_options import TestOptions

opt=TestOptions()
opt.initialize() #先只初始化,这样参数添加进去了
if device.type=='cpu':
    opt.parser.set_defaults(gpu_ids='-1') #如果是cpu得改默认
opt=opt.parse()
#opt.crop_size=224
start_epoch,epoch_iter=1,0
torch.nn.Module.dump_patches=True
SimSwap_model=create_model(opt) #初始化模型
#SimSwap_model.to(device)
SimSwap_model.eval()

sys.path.remove(SimSwap_path) #移除更新的模块导入目录
os.chdir(work_path) #恢复工作目录

#SBI实例化模型,导入文件模块
os.chdir(SBI_path) #更新工作目录
sys.path.append(SBI_path+'\\src\\utils') #更新模块导入目录

from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
import blend as B
if os.path.isfile(SBI_path+'\\src\\utils\\library\\bi_online_generation.py'):
    sys.path.append(SBI_path+'\\src\\utils\\library\\')
    print('exist library')
    exist_bi=True			#使用第三方的Face X-Ray凸包生成程序,SBI检测性能有很大提升
else:
    exist_bi=False
if exist_bi:
    from library.bi_online_generation import random_get_hull
    sys.path.remove(SBI_path+'\\src\\utils\\library\\')

face_detector=dlib.get_frontal_face_detector()                        #实例化人脸检测类
predictor_path='.\\src\\preprocess\\shape_predictor_81_face_landmarks.dat' #人脸检测模型文件位置
print(predictor_path)
face_predictor=dlib.shape_predictor(predictor_path)                   #加载人脸landmarks检测模型

sys.path.remove(SBI_path+'\\src\\utils') #移除更新的模块导入目录
os.chdir(work_path) #恢复工作目录

##生成数据集

#用到的一些变换
#SimSwap里对图片做的预处理变换
transformer=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

transformer_Arcface=transforms.Compose([
        transforms.ToTensor(),           
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])              #调用Arcface前使用的Transformer

#SBI
#SBI,对原图片的变换
def get_source_transforms(): #对应于论文中Source-Target Generator使用的各种图像几何、频域变换
    """
        SBI,对原图片的变换
    """
    return alb.Compose([
            alb.Compose([
                    alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3), #按照参数指定的范围随机选择RGB三通道随机偏移值并对图像3个通道进行偏移（需要约束在图像合理范围内）
                    alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3),sat_shift_limit=(-0.3,0.3),val_shift_limit=(-0.3,0.3),p=1), #首先将图像转换为HSV格式,然后按照参数指定范围随机选择HSV偏移值并对H、S、V三通道进行偏移,执行约束后再转换为图像RGB格式
                    alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=1), #对图像的亮度、对比度进行随机偏移
                ],p=1),

            alb.OneOf([
                RandomDownScale(p=1), #图像缩放（先缩小再恢复原始尺度）
                alb.Sharpen(alpha=(0.2,0.5),lightness=(0.5,1.0),p=1), #图像锐化	
            ],p=1),
            
        ],p=1.)
source_transforms=get_source_transforms() # 代表论文中的def T (I)函数,执行颜色变换及频域变换

#总的函数
def generate_SSSBIDataset(input_path,output_path,already_folder_num=0,folder_num=10):
    """
        生成数据集

        :param input_path: 输入文件路径,包含若干子文件夹,子文件夹中包含被处理图片
        :param outputpath: 输出文件路径
        :param already_folder_num: 已处理的子文件夹个数,跳过
        :param folder_num: 处理的子文件夹个数(不包括跳过的)
    """    
    
    #创建输出文件夹(若不存在)
    os.makedirs(output_path,exist_ok=True)

    #遍历folder_num子文件夹
    #in_folder_list=sorted(glob(input_path+'*\\')) #获取所有子文件夹
    in_folder_list=sorted(glob(input_path + '*\\'))[already_folder_num:] #跳过已处理子文件夹
    in_folder_list=in_folder_list[:folder_num] #限制处理子文件夹总数
    for in_folder_path in in_folder_list:
        #print(f"Folder: {in_folder_path}")

        #创建对应的输出子文件夹
        out_folder_path=in_folder_path.replace(input_path,output_path,1) #一级输出文件夹
        os.makedirs(out_folder_path,exist_ok=True) #创建一级文件夹
        for j in range(4):
            os.makedirs(out_folder_path+str(j)+"\\",exist_ok=True) #创建四个二级文件夹,对应4种类别

        #遍历所有图片
        image_list=sorted(glob(os.path.join(in_folder_path,'*.jpg'))) #获取该子文件夹的所有图片
        print("{} images are exist in {}".format(len(image_list),in_folder_path)) #打印图片信息
        for img_path in tqdm(image_list):
            #print(f"Image: {img_path}")

            #读取,保存路径
            img_R_path=img_path.replace(input_path,output_path,1)
            img_S_path=os.path.dirname(img_R_path)+'\\0\\'+os.path.basename(img_R_path) #IS的保存路径
            img_SB_path=os.path.dirname(img_R_path)+'\\1\\'+os.path.basename(img_R_path) #ISB的保存路径
            img_SBR_path=os.path.dirname(img_R_path)+'\\2\\'+os.path.basename(img_R_path) #ISBR的保存路径
            img_R_path=os.path.dirname(img_R_path)+'\\3\\'+os.path.basename(img_R_path) #IR的保存路径
            
            img_PIL=Image.open(img_path).convert('RGB') #用PIL格式的图片,RGB
            img_np=np.array(img_PIL) #用PIL加载的numpy格式的图片,RGB
            img_cv=cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR) #用opencv格式的图片,BGR
            
            #生成IS
            #即真实图片,对原始图片随机进行一种变换
            img_S=random_augmentation(img_cv) #生成IS

            #生成IR
            with torch.no_grad():
                #numpy转换为tensor
                img_s=transformer_Arcface(img_PIL) #源图片
                img_id=img_s.view(-1,img_s.shape[0],img_s.shape[1],img_s.shape[2])
                img_t=transformer(img_PIL) #目标图片
                img_att=img_t.view(-1,img_t.shape[0],img_t.shape[1],img_t.shape[2])

                #张量迁移到设备
                img_id=img_id.to(device)
                img_att=img_att.to(device)

                #创建身份ID
                img_id_downsample=F.interpolate(img_id,size=(112,112)) #下采样,arcface模型输入大小
                latend_id=SimSwap_model.netArc(img_id_downsample) #Inference并获得身份ID
                latend_id=latend_id.detach().to('cpu')
                latend_id=latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True) #身份ID归一化
                latend_id=latend_id.to(device)
                
                #生成IR
                img_R=SimSwap_model(img_id,img_att,latend_id,latend_id,True) #用SimSwap方法生成图像IR 得到的图像是张量(通道在前),0-1,RGB的

                #拼接图片 都先转换为opencv格式,方便与后面sbi统一
                img_R=(np.array(img_R[0].permute(1,2,0).to('cpu'))*255).astype(np.uint8) #IR
                
            #尝试生成ISB和ISBR
            #获取landmark和bbox 共用一种
            landmark_bbox=facecrop(img_cv,face_detector=face_detector,face_predictor=face_predictor)
            #未检测到人脸,直接进入下一图片
            if landmark_bbox==False:
                #print('No faces in {}'.format(img_path))
                tqdm.write(f'No faces in {img_path}')
                continue
            #检测到人脸则进行生成
            else:
                landmarks,bboxes=landmark_bbox

                img_SB,SB_blend_mask,landmark,bbox,mask_f=SBI(img_np,img_np,landmarks,bboxes) #自拼接生成图片,获取边界掩膜,原始掩膜
                SB_edit_mask=np.zeros_like(img_SB) #ISB的操纵掩膜,全0

                img_SBR,SBR_blend_mask,landmark,bbox,mask_f=SBI(img_R,img_np,landmarks,bboxes) #IR是source,IS是target生成图片,获取边界掩膜,原始掩膜
                non_zero_mask=np.any(mask_f!= 0, axis=-1)
                #SBR_edit_mask_Braces=np.where(non_zero_mask[..., np.newaxis], 255, mask_f) #ISBR的操纵掩膜,原始掩膜非0的地方全为255 {0,1}
                #SBR_edit_mask_Parentheses=mask_f #操纵掩膜（0,1）就行
                #SBR_edit_mask=mask_f #ISBR的操纵掩膜,(0,1)
                SBR_edit_mask=np.where(non_zero_mask[..., np.newaxis], 255, mask_f) #ISBR的操纵掩膜,{0,1}
            
            #保存
            #保存IS
            img_S,_,__,___=crop_face(img_S,landmark,bbox,margin=True,crop_by_bbox=False)
            img_S,_,__,___=crop_face(img_S,landmark,bbox,margin=False,crop_by_bbox=True,phase='train') #类似SBI中两次裁剪
            img_S=cv2.resize(img_S,image_size,interpolation=cv2.INTER_LINEAR).astype(np.uint8)	#样本缩放为原始尺寸
            S_blend_mask=np.zeros_like(img_S) #IS的边界掩膜,全0
            S_edit_mask=np.zeros_like(img_S) #IS的操作掩膜,全0

            img_S_full=np.concatenate([img_S,S_blend_mask,S_edit_mask],axis=1) #拼接IS

            img_S_full_tensor=img_S_full.transpose((2,0,1)).astype('float32')/255 #改为通道在前,值为0-1
            img_S_full_tensor=torch.tensor(img_S_full_tensor).float() #变张量
            utils.save_image(img_S_full_tensor, img_S_path, normalize=False, value_range=(0, 1)) #保存IS

            #保存IR
            img_R,_,__,___=crop_face(img_R,landmark,bbox,margin=True,crop_by_bbox=False)
            img_R,_,__,___=crop_face(img_R,landmark,bbox,margin=False,crop_by_bbox=True,phase='train') #类似SBI中两次裁剪
            img_R=cv2.resize(img_R,image_size,interpolation=cv2.INTER_LINEAR).astype(np.uint8)	#样本缩放为原始尺寸
            R_blend_mask=np.zeros_like(img_R) #IR的边界掩膜,全0
            R_edit_mask=np.full_like(img_R,255) #IR的操纵掩膜,全1 整个图片全来自GAN

            img_R_full=np.concatenate([img_R,R_blend_mask,R_edit_mask],axis=1) #拼接图片
            
            img_R_full_tensor=img_R_full.transpose((2,0,1)).astype('float32')/255 #改为通道在前,值为0-1
            img_R_full_tensor=torch.tensor(img_R_full_tensor).float() #变张量
            utils.save_image(img_R_full_tensor, img_R_path, normalize=False, value_range=(0, 1)) #保存IR

            #保存ISB
            img_SB_full=np.concatenate([img_SB,SB_blend_mask,SB_edit_mask],axis=1) #拼接ISB

            img_SB_full_tensor=img_SB_full.transpose((2,0,1)).astype('float32')/255 #改为通道在前,值为0-1
            img_SB_full_tensor=torch.tensor(img_SB_full_tensor).float() #变张量
            utils.save_image(img_SB_full_tensor, img_SB_path, normalize=False, value_range=(0, 1)) #保存ISB

            #保存ISBR
            img_SBR_full=np.concatenate([img_SBR,SBR_blend_mask,SBR_edit_mask],axis=1) #拼接ISBR

            img_SBR_full_tensor=img_SBR_full.transpose((2,0,1)).astype('float32')/255 #改为通道在前,值为0-1
            img_SBR_full_tensor=torch.tensor(img_SBR_full_tensor).float() #变张量
            utils.save_image(img_SBR_full_tensor, img_SBR_path, normalize=False, value_range=(0, 1)) #保存ISBR

    print("All done!") #打印完成信息

#生成IS所用,图像增强操作
def random_augmentation(img): #接受opencv格式图片,返回PIL读取的numpy格式图片
    """
        生成IS所用,图像增强操作

        :param img: 待处理文件,opencv格式
        :return img_S: IS,PIL读取的numpy格式
    """
    #选择一种处理方法
    choice=random.choice(['original','jpeg_compression','color_distortion','brightness_contrast','gaussian_blur']) #对图片随机进行空处理、随机JPEG压缩、颜色扰动、对比度调节、高斯滤波

    #空处理
    if choice=='original':
        img_S=img

    #随机JPEG压缩
    elif choice=='jpeg_compression':
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),random.randint(50,95)]
        _,img_encode=cv2.imencode('.jpg',img,encode_param)
        img_S=cv2.imdecode(img_encode,1)

    #颜色扰动
    elif choice=='color_distortion':
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsv[:,:,0]=(hsv[:,:,0] + random.randint(-10,10))%180  #调整色调
        hsv[:,:,1]=np.clip(hsv[:,:,1] * random.uniform(0.8,1.2),0,255)  #调整饱和度
        hsv[:,:,2]=np.clip(hsv[:,:,2] * random.uniform(0.8,1.2),0,255)  #调整亮度
        img_S=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    #对比度调节
    elif choice=='brightness_contrast':
        alpha=random.uniform(0.8,1.2)  #对比度调节系数
        beta=random.randint(-20,20)    #亮度调节值
        img_S=cv2.convertScaleAbs(img,alpha=alpha,beta=beta)

    #高斯滤波
    elif choice=='gaussian_blur':
        img_S=cv2.GaussianBlur(img,(5,5),0)

    img_S=cv2.cvtColor(img_S,cv2.COLOR_BGR2RGB) #转换回RGB格式

    return img_S

#SBI 生成获取图片landmark,bbox
def facecrop(img_org,face_detector,face_predictor):
    """
        SBI,生成获取图片landmark,bbox

        :param img_org: 原图片
        :param face_detector: 人脸bbox检测模型
        :param face_predictor: 人脸landmark检测模型
        :return landmarks: 检测到的landmark列表
        :return bboxes: 检测到的bboxes列表
    """
    img=cv2.cvtColor(img_org,cv2.COLOR_BGR2RGB)

    faces=face_detector(img,1) #检测人脸bounding box
    if len(faces)==0:
        return False #若没有检测到,直接返回false

    landmarks=[] #初始化人脸landmarks列表
    size_list=[] #初始化人脸landmarks最小外接矩形像素面积列表
    bboxes=[] #初始化boundingbox
    for face_idx in range(len(faces)):
        landmark=face_predictor(img,faces[face_idx])   #获取人脸landmark
        landmark=face_utils.shape_to_np(landmark)         #将landmark转换为numpy数组
        x0,y0=landmark[:,0].min(),landmark[:,1].min()       #获取人脸landmark最小外接矩形左上角坐标
        x1,y1=landmark[:,0].max(),landmark[:,1].max()       #获取人脸landmark最小外接矩形右下角坐标
        face_s=(x1-x0)*(y1-y0)                              #计算人脸landmark最小外接矩形像素面积
        size_list.append(face_s)                            #存入人脸landmark最小外接矩形像素面积列表
        landmarks.append(landmark)                          #存入人脸landmark列表
        bboxes.append([[faces[face_idx].left(),faces[face_idx].top()],[faces[face_idx].right(),faces[face_idx].bottom()]])  #存入人脸boundingbox列表

    landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)   #将一帧内所有人脸的landmark合并为一个数组,并增加一个人脸索引维度
    landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]  #将人脸landmarks数组按照landmarks最小外接矩形像素面积倒序排序(目的是仅取最大面积的人脸)
    
    bboxes=np.array(bboxes,dtype=np.int32)
    bboxes=bboxes[np.argsort(np.array(size_list))[::-1]]    #将人脸boundingbox数组按照landmarks最小外接矩形像素面积倒序排序(目的是仅取最大面积的人脸)

    return landmarks,bboxes
    
#SBI,生成SBI
def SBI(img_source,img_target,landmarks,bboxes): #landmark、bboxes是source的,如果是ISBR,IR是source,IS是target
    """
        SBI,生成SBI

        :param img_source: 源图片,脸部分
        :param img_target: 目标图片,脸以外
        :param landmarks: 源人脸的landmark,和目标人脸共用
        :param bboxes: 源人脸的bboxe,和目标人脸共用
        :return img_f: 自拼接图片
        :return blend_mask: 边界掩膜
        :return landmark: 对应的landmark
        :return bbox: 对应的bbox
        :return mask_f: 原始掩膜
    """
    landmark=landmarks[0]
    bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])	#获取landmarks外接最小矩形框的左上角及右下角坐标
    bboxes=bboxes[:2]
    iou_max=-1								#IoU最大值
    for i in range(len(bboxes)):
        iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())	#计算bbox_lm与bboxes之间的IoU
        if iou_max<iou:
            bbox=bboxes[i]								#记录IoU最大的boundingbox
            iou_max=iou									#记录当前最大IoU
    
    landmark=reorder_landmark(landmark)			#landmarks重新排序,主要调整了前额(68~80)序号的数组顺序

    #第一次图像剪切    
    #img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)	#图像剪切 
    img_source,landmark,bbox,__,y0_new,y1_new,x0_new,x1_new=crop_face(img_source,landmark,bbox,margin=True,crop_by_bbox=False,abs_coord=True)	#图像剪切 
    img_target=img_target[y0_new:y1_new,x0_new:x1_new] #对targe也做相同裁剪

    #自拼接
    img_r,img_f,mask_f,blend_ratio=self_blending(img_source.copy(),img_target.copy(),landmark.copy())		#人脸拼接,内包含对图片、掩码进行几何及频域变换以增强数据

    #第二次图像剪切,会造成大小变化
    #img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)	#图像剪切增强,注意假样本剪切的边框算法选择了与帧样本不同的方法,主要是为了增加分散性			    
    img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase='train')	#图像剪切增强,注意假样本剪切的边框算法选择了与帧样本不同的方法,主要是为了增加分散性
    mask_f=mask_f[y0_new:y1_new,x0_new:x1_new] 

    #img_r=img_r[y0_new:y1_new,x0_new:x1_new]	#img_f代表假样本、img_r代表真样本,对真样本用假样本的剪切边框重新剪切真样本 
            
    #img_f=cv2.resize(img_f,image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255	#样本缩放为原始尺寸
    #img_r=cv2.resize(img_r,image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                

    #img_f=img_f.transpose((2,0,1))		#图像格式改为通道在前
    #img_r=img_r.transpose((2,0,1))
    #flag=False  

    #用xray的方式计算拼接边界
    mask=mask_f/blend_ratio #中间全白的mask
    mask_oppo=1-mask
    blend_mask=mask*mask_oppo*4 #x-ray的公式 得到的是单通道图,0-1

    img_f=cv2.resize(img_f,image_size,interpolation=cv2.INTER_LINEAR).astype(np.uint8)	#样本缩放为原始尺寸

    blend_mask=(cv2.resize(blend_mask,image_size,interpolation=cv2.INTER_LINEAR)*255).astype(np.uint8) #掩膜缩放回原始尺寸
    blend_mask=np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2) #单通道变三通道,便于拼接

    mask_f=(cv2.resize(mask_f,image_size,interpolation=cv2.INTER_LINEAR)*255).astype(np.uint8) #掩膜缩放回原始尺寸
    mask_f=np.repeat(mask_f[:, :, np.newaxis], 3, axis=2) #单通道变三通道,便于拼接

    return img_f,blend_mask,landmark,bbox,mask_f #返回的是通道在最后,值为0-255,RGB的图片

#SBI,landmark重新排序
def reorder_landmark(landmark):             #对landmarks重新排序,81个Landmarks比68个landmarks多了13个landmarks,主要分布在前额,
    """
        SBI,landmark重新排序
    """
    landmark_add=np.zeros((13,2))			#前额landmarks数组,对应于81个landmarks的第68~80序号
    for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
        landmark_add[idx]=landmark[idx_l]	#重新排序68以后的landmark,原77号（正脸左前额角,左眼角左上方）,对应于新的68号,原78号（正脸右前额角,右眼角右上方）对应于新的80号
    landmark[68:]=landmark_add
    return landmark
    
#SBI,人脸拼接
#def self_blending(img,landmark):							#用图片及landmarks进行自拼接
def self_blending(img_source,img_target,landmark):          #用target和source自拼接,landmark是source的
    """
        SBI,人脸拼接

        :param img_source: 源图片,脸部分
        :param img_target: 目标图片,脸以外
        :param landmark: 源人脸的landmark,和目标人脸共用
        :return img_target: 目标图片
        :return img_blended: 自拼接后的图片
        :return mask: 拼接掩膜
        :return blend_ratio: 拼接混合率
    """
    H,W=len(img_source),len(img_source[0])
    if np.random.rand()<0.25:									#随机选择68/81个landmarks的mask生成方案
        landmark=landmark[:68]
    if exist_bi:
        logging.disable(logging.FATAL)
        mask=random_get_hull(landmark,img_source)[:,:,0]				#调用Face X-Ray函数计算mask
        logging.disable(logging.NOTSET)
    else:
        mask=np.zeros_like(img_source[:,:,0])							#初始化mask
        cv2.fillConvexPoly(mask,cv2.convexHull(landmark),1.)	#用landmarks计算凸包并计算mask

    #source=img.copy()											#由于是自拼接,所以source、img分别代表论文里的source、target
    if np.random.rand()<0.5:									#随机对图片进行source_transforms变换
        #source=source_transforms(image=source.astype(np.uint8))['image']
        img_source=source_transforms(image=img_source.astype(np.uint8))['image']
    else:
        img_target=source_transforms(image=img_target.astype(np.uint8))['image']
    
    img_source,mask=randaffine(img_source,mask)					#对source、mask进行随机仿射变换及弹性变换（数据增强）

    img_blended,mask,blend_ratio=B.dynamic_blend(img_source,img_target,mask)			#利用mask对source、img执行动态拼接
    img_blended=img_blended.astype(np.uint8)
    img_target=img_target.astype(np.uint8)

    return img_target,img_blended,mask,blend_ratio
    
#SBI,仿射变换
def randaffine(img,mask):				#对图像及掩码同时执行一致的仿射变换、弹性变换
    """
        SBI,仿射变换

        :param img: 源图片,脸部分
        :param mask: 掩膜
    """
    f=alb.Affine(
            translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},	#平移
            scale=[0.95,1/0.95],										#缩放
            fit_output=False,
            p=1)
    
    g=alb.ElasticTransform(
            alpha=50,													#高斯核尺寸
            sigma=7,													#高斯核方差
            alpha_affine=0,
            p=1,
        )
    
    transformed=f(image=img,mask=mask)									#先进行仿射变换
    img=transformed['image']
    
    mask=transformed['mask']							
    transformed=g(image=img,mask=mask)									#再进行弹性变换
    mask=transformed['mask']
    return img,mask

##生成数据集
#开始计时
start_time=time.time()

generate_SSSBIDataset(input_path,output_path,already_folder_num,folder_num)  #这里可以改生成的文件夹数

#结束计时
end_time=time.time()
run_time=end_time - start_time
print("程序运行时间为 {:.2f} 秒".format(run_time))
