import os
import sys
current_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_pwd}/../src/')
DEFAULT_DATA_PATH = f'{current_pwd}/../data/test_data/'

import glob
import os
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def crop_image(img, mask):
    (x,y,w,h) = cv2.boundingRect(np.uint8(mask))
    mask = mask[y:y+h,x:x+w]
    img = img[y:y+h,x:x+w]

    # Prepare new image, with correct size:
    margin = 1.1
    im_size = 256
    
    size = int((max(w, h)*margin)//2)
    new_x = size - w//2
    new_y = size - h//2
    new_img = np.zeros((size*2, size*2, 3))
    new_mask = np.zeros((size*2, size*2))
    new_img[new_y:new_y + h, new_x:new_x+w] = img
    new_mask[new_y:new_y + h, new_x:new_x+w] = mask

    # Resizing cropped and centered image to desired size:
    img = cv2.resize(new_img, (im_size,im_size))
    mask = cv2.resize(new_mask, (im_size,im_size))

    offset_x = x - new_x
    offset_y = y - new_y
    

    return img, mask, (offset_x,offset_y), (size*2)/im_size

def vid2frames(video_file, out_clip_dir, fps_out=60):
    
    # ---------------------
    # Read frames: front and side view
    # ---------------------
    ys = YOLO("yolov8m-seg.pt")
    vidcap = cv2.VideoCapture(video_file)
    fps_in = vidcap.get(cv2.CAP_PROP_FPS)

    #fps_out = 60
    assert fps_in == fps_in, 'testing data should all be 60 fps'

    success_front, image_front = vidcap.read()
    
    sampled_idx = []
    frame_idx_in = -1
    frame_cnt = -1
    while success_front == True:
        # check if sample
        frame_idx_in += 1
        ratio = fps_in / fps_out
        frame_idx_out = int(frame_idx_in / ratio)
        
        #print(frame_idx_out, frame_cnt, frame_idx_in)
        
        if frame_idx_out > frame_cnt:
            frame_cnt += 1
            sampled_idx.append(frame_idx_in)
            cv2.imwrite(os.path.join(out_clip_dir, f'frame_{frame_cnt}_org.png'), image_front)
            
            results = ys.predict(image_front[:,:,::-1], verbose = False)
            for result in results:
                # iterate results
                if result.boxes is not None and result.masks is not None:
                    boxes = result.boxes.cpu().numpy().data   
                    masks = result.masks.cpu().numpy().data# get boxes on cpu in numpy
                    # find the bod with person with highest score
                    # extract classes
                    clss = boxes[:, 5]
                    # get indices of results where class is 0 (people in COCO)
                    people_indices = []
                    for i, cls in enumerate(clss):
                        if result.names[cls] == 'person':
                            people_indices.append(i)
                    # use these indices to extract the relevant masks
                    
                    selected_idx = -1
                    conf = -1000
                    for idx in people_indices: 
                        if boxes[idx, 4] > conf:
                            selected_idx = idx
                            conf = boxes[idx, 4] 
                    
                    masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
                    # rescale masks to original image
                    masks = scale_image(masks, result.masks.orig_shape)
                    masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)
                            
                    mask = (masks[selected_idx]>0.2) * 255
                    
                    img, mask, (off_x,off_y), scales = crop_image(image_front, mask)
                            
                    cv2.imwrite(os.path.join(out_clip_dir, f'frame_{frame_cnt}_mask.png'), mask )
                    cv2.imwrite(os.path.join(out_clip_dir, f'frame_{frame_cnt}_crop.png'), img)
                    # save meta
                    meta = [off_x, off_y, scales]
                    np.save(os.path.join(out_clip_dir, f'frame_{frame_cnt}_meta.npy'), meta)
                
                            
                
        success_front, image_front = vidcap.read()
    
    vidcap.release()
    return sampled_idx
                        
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_PATH, help="Specify data_dir")    
    parser.add_argument('--out_dir', type=str, default=DEFAULT_DATA_PATH, help="Specify out_dir")   
    
    args = parser.parse_args()   
    
    dataset = 'ODAH'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_list = glob.glob(os.path.join(args.data_dir, dataset, 'video', '*/*_front.mp4'))
    os.makedirs(args.out_dir, exist_ok=True)

    for clip in video_list:
            
        subject = str.split(clip, '/')[-2]
        
        fps_out = 60
        action = str.split(str.split(clip, '/')[-1], '_')[0]
        subject = str.split(clip, '/')[-2]
        
        print(f'-------------------{subject}--------------------')
        print('subject:', subject, ', action:', action)
        
        ## save frames from video ...
        img_out_dir1 = os.path.join(args.out_dir, dataset, 'images', subject, action, 'view0')
        if os.path.exists(img_out_dir1) is False:
            os.makedirs(img_out_dir1, exist_ok=True)
            sampled_idx_view0 = vid2frames(clip, img_out_dir1, fps_out)
            
            clip_view2 =clip.replace('_front.mp4', '_side.mp4')
                
            img_out_dir2 = os.path.join(args.out_dir, dataset, 'images', subject, action, 'view1')
            os.makedirs(img_out_dir2, exist_ok=True)
            sampled_idx_view1 = vid2frames(clip_view2, img_out_dir2, fps_out)  
