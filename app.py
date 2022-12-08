import matplotlib.cm as cm
import torch
#import gradio as gr
from models.matching import Matching
from models.utils import (make_matching_plot_fast, process_image)
import cv2

torch.set_grad_enabled(False)

# Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

resize = [640, 640]
max_keypoints = 1024
keypoint_threshold = 0.005
nms_radius = 4
sinkhorn_iterations = 20
match_threshold = 0.2
resize_float = False

config_indoor = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_keypoints': max_keypoints
    },
    'superglue': {
        'weights': "indoor",
        'sinkhorn_iterations': sinkhorn_iterations,
        'match_threshold': match_threshold,
    }
}

config_outdoor = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_keypoints': max_keypoints
    },
    'superglue': {
        'weights': "outdoor",
        'sinkhorn_iterations': sinkhorn_iterations,
        'match_threshold': match_threshold,
    }
}

matching_indoor = Matching(config_indoor).eval().to(device)
matching_outdoor = Matching(config_outdoor).eval().to(device)

def run(input0, input1, superglue):
    if superglue == "indoor":
        matching = matching_indoor
    else:
        matching = matching_outdoor
    
    name0 = input0
    name1 = input1

    # If a rotation integer is provided (e.g. from EXIF data), use it:
    rot0, rot1 = 0, 0

    # Load the image pair.
    image0, inp0, scales0 = process_image(input0, device, resize, rot0, resize_float)
    image1, inp1, scales1 = process_image(input1, device, resize, rot1, resize_float)

    if image0 is None or image1 is None:
        print('Problem reading image pair')
        return

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
            
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

   
    # Visualize the matches.
    color = cm.jet(mconf)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        '{}'.format(len(mkpts0)),
    ]

    if rot0 != 0 or rot1 != 0:
        text.append('Rotation: {}:{}'.format(rot0, rot1))

    # Display extra parameter info.
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {}:{}'.format(name0, name1),
    ]

    output = make_matching_plot_fast(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
        text, show_keypoints=True, small_text=small_text)

    print('Source Image - {}, Destination Image - {}, {}, Match Percentage - {}'.format(name0, name1, text[2], len(mkpts0)/len(kpts0)))
    return output, text[2], str((len(mkpts0)/len(kpts0))*100.0) + '%'

if __name__ == '__main__':

    # glue = gr.Interface(
    #     fn=run, 
    #     inputs=[
    #         gr.Image(label='Input Image'),
    #         gr.Image(label='Match Image'),
    #         gr.Radio(choices=["indoor", "outdoor"], value="indoor", type="value", label="SuperGlueType", interactive=True),
    #     ], 
    #     outputs=[gr.Image(
    #         type="pil",
    #         label="Result"),
    #         gr.Textbox(label="Keypoints Matched"),
    #         gr.Textbox(label="Match Percentage")
    #     ],
    #     examples=[
    #         ['./taj-1.jpg', './taj-2.jpg', "outdoor"],
    #         ['./outdoor-1.JPEG', './outdoor-2.JPEG', "outdoor"]
    #     ]    
    # )
    # glue.queue()
    # glue.launch()
    count = 0
    firstSet = ['realWatch1.png', 'realWatch2.png', 'realWatch3.png', 'realWatch4.png']
    secondSet = ['digitalWatch1.png', 'digitalWatch2.png','digitalWatch3.png']
    for first in firstSet:
        for second in secondSet:
            output, _, __ = run(first, second,"outdoor")
            cv2.imwrite('out'+ str(count) +'.png', output)
            count += 1
    firstSet = ['realShoe1.png', 'realShoe2.png']
    secondSet = ['digitalShoe1.png', 'digitalShoe2.png','digitalShoe3.png']
    for first in firstSet:
        for second in secondSet:
            output, _, __ = run(first, second,"outdoor")
            cv2.imwrite('out'+ str(count) +'.png', output)
            count += 1
    firstSet = ['realBurger1.JPG', 'realBurger2.JPG']
    secondSet = ['digitalBurger1.png', 'digitalBurger2.png','digitalBurger3.png']
    for first in firstSet:
        for second in secondSet:
            output, _, __ = run(first, second,"outdoor")
            cv2.imwrite('out'+ str(count) +'.png', output)
            count += 1
    