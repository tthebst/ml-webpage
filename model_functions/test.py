import matplotlib.pyplot as plt
import json
import urllib.request
import torchvision
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tensorflow as tf
import tensorflow_hub as hub
url, filename = ("https://www.lausanne2020.sport/var/ezdemo_site/storage/images/media/news-story-galleries/18_01_ice_hockey/kehl/640454-1-eng-GB/kehl_col12Width.jpg", "dog.jpg")
try:
    urllib.request.urlretrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)
pred = [4.057498514953295e-08, 5.984022610761031e-09, 6.392280482714341e-09, 2.759398487484077e-09, 4.34829505735479e-09, 4.089483418567852e-09, 3.7821596379217226e-06, 8.37522939889368e-09, 9.992549898640846e-09, 4.5091290701293474e-08, 6.166311350419562e-10, 5.183088180871209e-09, 1.531843309976466e-07, 5.475886410977182e-08, 1.7676883601325244e-07, 2.564276257999154e-07, 3.3770996310522605e-07, 1.1143817424397184e-08, 1.575803487696703e-09, 4.9049216244156923e-08, 5.711492576665478e-07, 8.28495316795852e-09, 1.7346341030233248e-09, 2.9610802698698535e-09, 2.7436765748234393e-08, 5.015335204916482e-07, 1.2224094803059415e-07, 5.601892727469249e-09, 8.835196574352722e-09, 4.5241335122625514e-09, 2.0733043726295364e-08, 2.7104759325879968e-08, 4.7835988503663884e-09, 1.5041106138369287e-08, 5.0160789299980024e-08, 6.240194494466778e-08, 6.2928293687036785e-09, 1.8957104330752372e-08,
        1.6208726094646408e-07, 5.479686482345869e-09, 8.912767412994071e-09, 4.303949696637943e-10, 7.0130519169708805e-09, 4.825931299023978e-08, 9.558023261746484e-10, 5.964153615423129e-08, 3.160145567449035e-08, 1.2120432302253903e-07, 3.443920491008612e-07, 2.4218218541705028e-08, 2.115813080294515e-09, 6.639865546276269e-08, 6.879559464323393e-07, 1.427895579553251e-08, 1.6187314599847014e-07, 2.5770130918090217e-08, 3.481594745835537e-08, 4.026775002330396e-08, 8.507935689294754e-08, 2.360758770691973e-07, 1.833044827037611e-08, 3.290691807933399e-08, 1.5053934987463435e-08, 8.140016660718175e-09, 1.515578702537823e-07, 2.3133870374891785e-09, 1.716300701559703e-08, 2.0576917947323636e-08, 6.308058431159225e-08, 4.485026749989629e-08, 2.4505306117639236e-10, 1.2101692092869598e-08, 3.929917724576626e-08, 1.2409415717229422e-08, 4.115022544937119e-09, 1.3263321818612894e-07, 9.335192885373544e-07]


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], fill=False, edgecolor=color, linewidth=2)


input_image = Image.open(filename)


width, height = input_image.size[:2]
new_height = 500
new_width = new_height * width / height
input_image = input_image.resize((int(new_width), new_height), Image.ANTIALIAS)


print(type(input_image))
tarans = torchvision.transforms.ToTensor()
print(type(input_image))
input_image = tarans(input_image)
print(type(input_image))
print(input_image.shape)


input_batch = input_image.unsqueeze(0)


detector = hub.Module("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")

detector_output = detector(tf.convert_to_tensor(input_batch.cpu().detach().numpy()), as_dict=True)

print(detector_output)
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# For training
# For inference
# model.eval()
image = input_batch
predictions = model(image)
print(predictions)
image = [np.moveaxis(image[0].cpu().detach().numpy(), 0, -1)]

fig = plt.imshow(image[0])

print(predictions[0]['boxes'].data.cpu().detach().numpy().tolist())
for boxes in predictions[0]['boxes'].data.cpu().detach().numpy().tolist():
    fig.axes.add_patch(bbox_to_rect(boxes, 'blue'))


buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
buf.close()
