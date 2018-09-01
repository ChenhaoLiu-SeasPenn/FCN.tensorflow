import os
import numpy as np
from PIL import Image, ImageOps
from skimage.transform import resize
from skimage.measure import block_reduce


def converter(r_path):
  in_path = r_path + 'annotation'
  out_path = r_path + 'annotations'
  bin_path = r_path + 'annotations_bin'

  #     in_path = './amps_train_21/annotation'
  #     out_path = './amps_train_21/annotations'
  #     bin_path = './amps_train_21/annotations_bin'

  if not os.path.exists(out_path):
    os.mkdir(out_path)
    os.mkdir(bin_path)

  colors = {'trees': np.array([197, 180, 178]), 'houses': np.array([214, 178, 228]),
            'road': np.array([178, 208, 228]), 'Amps': np.array([228, 178, 178]),
            'Amps_overlay': np.array([233, 143, 143]), 'houses_overlay': np.array([208, 143, 233]),
            'tree_on_road': np.array([162, 169, 183])}
  # colors = {'trees': np.array([197, 180, 178]), 'houses': np.array([214, 178, 228]),
  #           'road': np.array([178, 208, 228]),  'houses_overlay': np.array([208, 143, 233]),
  #          'tree_on_road': np.array([162, 169, 183])}
  colors_lbl = {'trees': np.array([255, 0, 0]), 'houses': np.array([0, 255, 0]),
                'road': np.array([0, 0, 255]), 'Amps': np.array([255, 255, 0]),
                'Amps_overlay': np.array([255, 255, 0]), 'houses_overlay': np.array([0, 255, 0]),
                'tree_on_road': np.array([255, 0, 0])}
  # colors_lbl = {'trees': np.array([255, 0, 0]), 'houses': np.array([0, 255, 0]),
  #               'road': np.array([0, 0, 255]), 'houses_overlay': np.array([0, 255, 0]),
  #              'tree_on_road': np.array([255, 0, 0])}
  bin_lbl = {'trees': 1, 'houses': 2, 'road': 3, 'Amps': 4, 'Amps_overlay': 4,
             'houses_overlay': 2, 'tree_on_road': 1}
  # bin_lbl ={'trees': 1, 'houses': 2, 'road': 3,
  #           'houses_overlay': 2, 'tree_on_road': 1}

  in_img = os.listdir(in_path)

  for file in in_img:

    if os.path.isfile(os.path.join(in_path, file)):

      label = np.asarray(Image.open(os.path.join(in_path, file)))
      label = label[..., :-1]

      label_bin = np.zeros(label.shape[:-1])
      label_color = np.zeros_like(label)
      for type_lbl, _ in colors.items():
        diff = np.sum(np.abs(label - colors[type_lbl].reshape(1, 1, -1)), axis=2)
        label_bin[np.where(diff < 10)] = bin_lbl[type_lbl]
        label_color[np.where(diff < 10)] = colors_lbl[type_lbl].reshape(1, 1, -1)

      # print np.max(label_bin)

      label_handle = Image.fromarray(label_bin.astype(np.uint8))
      label_handle2 = Image.fromarray(label_color.astype(np.uint8))
      label_handle.save(os.path.join(out_path, file))
      label_handle2.save(os.path.join(bin_path, file))


def randomCropper():
  rootPath = '../data/multiclass/trainset/'
  #   rootPath = './amps_train_21/'
  imgPath = rootPath + 'image'
  annoPath = rootPath + 'annotations'
  imgOutPath = rootPath + 'train/images/training/'
  annoOutPath = rootPath + 'train/annotations/training/label_'
  annocOutPath = rootPath + 'train/annotations_patch_classify/training/label_'
  imgValOutPath = rootPath + 'train/images/validation/'
  annoValOutPath = rootPath + 'train/annotations/validation/label_'
  annocValOutPath = rootPath + 'train/annotations_patch_classify/validation/label_'

  if not os.path.exists(rootPath + 'train/'):
    os.mkdir(rootPath + 'train/')
    os.mkdir(rootPath + 'train/images/')
    os.mkdir(rootPath + 'train/annotations/')
    os.mkdir(rootPath + 'train/annotations_patch_classify/')
    os.mkdir(imgOutPath)
    os.mkdir(imgValOutPath)
    os.mkdir(annoOutPath[:-6])
    os.mkdir(annoValOutPath[:-6])
    os.mkdir(annocOutPath[:-6])
    os.mkdir(annocValOutPath[:-6])

  scales = [1, 1, 1, 1, 1, 1]

  cfile = open(rootPath + 'classification_data_train.txt', 'w')
  cfile_v = open(rootPath + 'classification_data_validation.txt', 'w')

  allImages = os.listdir(imgPath)
  cnt = 0
  for imgName in allImages:
    imgFile = os.path.join(imgPath, imgName)
    annoFile = os.path.join(annoPath, imgName[:-4] + '_label.png')

    if not os.path.isfile(imgFile):
      continue
    img = np.asarray(Image.open(imgFile))
    anno = np.asarray(Image.open(annoFile))
    level = int(imgName[:2])
    # print level

    for i in range(100 / (level - 18)):
      seedS = np.random.randint(6)
      seedX = np.random.randint(1024 - 256)
      seedY = np.random.randint(256)
      patchImg = img[seedY:seedY + 256 / scales[seedS], seedX:seedX + 256 / scales[seedS], ...]
      patchAnno = anno[seedY:seedY + 256 / scales[seedS], seedX:seedX + 256 / scales[seedS]]
      patchImg = resize(patchImg, (256, 256), mode='symmetric', preserve_range=True).astype(np.uint8)
      patchAnno = resize(patchAnno, (256, 256), mode='symmetric', preserve_range=True).astype(np.uint8)
      patchAnno = np.ceil(patchAnno).astype(np.uint8)
      anno_toSeg = (patchAnno == 4).astype(np.uint8)
      patchAnno[patchAnno == 4] = 0
      if np.max(anno_toSeg) > 0:
        print(np.max(anno_toSeg))
      anno_patchClassify = block_reduce(anno_toSeg, (64, 64), func=np.max)

      img_ts = Image.fromarray(patchImg)
      anno_ts = Image.fromarray(patchAnno)
      annoc_ts = Image.fromarray(anno_patchClassify)
      if cnt % 10 != 9:
        img_ts.save(imgOutPath + str(cnt) + '.png')
        ImageOps.flip(img_ts).save(imgOutPath + str(cnt) + '_f.png')
        ImageOps.mirror(img_ts).save(imgOutPath + str(cnt) + '_m.png')
        anno_ts.save(annoOutPath + str(cnt) + '.png')
        ImageOps.flip(anno_ts).save(annoOutPath + str(cnt) + '_f.png')
        ImageOps.mirror(anno_ts).save(annoOutPath + str(cnt) + '_m.png')
        annoc_ts.save(annocOutPath + str(cnt) + '.png')
        ImageOps.flip(annoc_ts).save(annocOutPath + str(cnt) + '_f.png')
        ImageOps.mirror(annoc_ts).save(annocOutPath + str(cnt) + '_m.png')
        if np.sum(np.int32(anno_patchClassify == 4)) > 0:
          cfile.write(imgOutPath + str(cnt) + '.png 1\n')
          cfile.write(imgOutPath + str(cnt) + '_f.png 1\n')
          cfile.write(imgOutPath + str(cnt) + '_m.png 1\n')
        else:
          cfile.write(imgOutPath + str(cnt) + '.png 0\n')
          cfile.write(imgOutPath + str(cnt) + '_f.png 0\n')
          cfile.write(imgOutPath + str(cnt) + '_m.png 0\n')
      else:
        img_ts.save(imgValOutPath + str(cnt) + '.png')
        ImageOps.flip(img_ts).save(imgValOutPath + str(cnt) + '_f.png')
        ImageOps.mirror(img_ts).save(imgValOutPath + str(cnt) + '_m.png')
        anno_ts.save(annoValOutPath + str(cnt) + '.png')
        ImageOps.flip(anno_ts).save(annoValOutPath + str(cnt) + '_f.png')
        ImageOps.mirror(anno_ts).save(annoValOutPath + str(cnt) + '_m.png')
        annoc_ts.save(annocValOutPath + str(cnt) + '.png')
        ImageOps.flip(annoc_ts).save(annocValOutPath + str(cnt) + '_f.png')
        ImageOps.mirror(annoc_ts).save(annocValOutPath + str(cnt) + '_m.png')
        if np.sum(np.int32(patchAnno == 1)) > 0:
          cfile_v.write(imgOutPath + str(cnt) + '.png 1\n')
          cfile_v.write(imgOutPath + str(cnt) + '_f.png 1\n')
          cfile_v.write(imgOutPath + str(cnt) + '_m.png 1\n')
        else:
          cfile_v.write(imgOutPath + str(cnt) + '.png 0\n')
          cfile_v.write(imgOutPath + str(cnt) + '_f.png 0\n')
          cfile_v.write(imgOutPath + str(cnt) + '_m.png 0\n')
      cnt += 1
      if cnt % 500 == 0:
        print(cnt)


def regularCropper():
  levels = [18, 19, 20, 21]

  for level in levels:
    rootPath = '../data/multiclass/test_' + str(level) + '/'
    #   rootPath = './amps_train_21/'
    imgPath = rootPath + 'image'
    annoPath = rootPath + 'annotations'
    imgOutPath = rootPath + 'test/images/training/'
    annoOutPath = rootPath + 'test/annotations/training/label_'
    annocOutPath = rootPath + 'test/annotations_patch_classify/training/label_'

    if not os.path.exists(rootPath + 'test/'):
      os.mkdir(rootPath + 'test/')
      os.mkdir(rootPath + 'test/images/')
      os.mkdir(rootPath + 'test/annotations/')
      os.mkdir(rootPath + 'test/annotations_patch_classify/')
      os.mkdir(imgOutPath)
      os.mkdir(annoOutPath[:-6])
      os.mkdir(annocOutPath[:-6])

    cfile = open(rootPath + 'classification_data_train.txt', 'w')
    scales = [1, 1, 1, 1, 1, 1]

    allImages = os.listdir(imgPath)
    cnt = 0
    for img in allImages:

      imgFile = os.path.join(imgPath, img)
      annoFile = os.path.join(annoPath, img[:-4] + '_label.png')

      if not os.path.isfile(imgFile):
        continue
      img = np.asarray(Image.open(imgFile))
      anno = np.asarray(Image.open(annoFile))

      for i in range(4):
        for j in range(2):
          patchImg = img[256 * j:256 * (j + 1), 256 * i:256 * (i + 1), ...]
          patchAnno = anno[256 * j:256 * (j + 1), 256 * i:256 * (i + 1)]
          patchImg = patchImg.astype(np.uint8)
          patchAnno = patchAnno.astype(np.uint8)
          # anno_toSeg = (patchAnno == 4).astype(np.uint8)
          patchAnno[patchAnno == 4] = 0
          # anno_patchClassify = block_reduce(anno_toSeg, (64, 64), func=np.max)

          img_ts = Image.fromarray(patchImg)
          anno_ts = Image.fromarray(patchAnno)
          # annoc_ts = Image.fromarray(anno_patchClassify)

          img_ts.save(imgOutPath + str(level) + '_' + str(cnt) + '.png')
          anno_ts.save(annoOutPath + str(level) +'_' + str(cnt) + '.png')
          # annoc_ts.save(annocOutPath + str(cnt) + '.png')

          # if np.sum(np.int32(anno_patchClassify == 1)) > 0:
          #   cfile.write(imgOutPath + str(cnt) + '.png 1\n')
          # else:
          #   cfile.write(imgOutPath + str(cnt) + '.png 0\n')

          cnt += 1
          if cnt % 500 == 0:
            print(cnt)

if __name__ == '__main__':
  # converter('../data/multiclass/trainset/')
  levels = [18, 19, 20, 21]
  for level in levels:
      converter('../data/multiclass/test_'+str(level) + '/')
  # randomCropper()
  regularCropper()