import argparse
import cv2
import glob
import os
import io
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from flask import Flask, request, render_template, redirect
from PIL import Image
import base64

# Use Real-ESRGAN
class esrgan():
    def __init__(self, scale):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image')
        parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
        parser.add_argument('-t', '--tile', type=int, default=256, help='Tile size, 0 for no tile during testing')
        parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
        parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
        parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
        parser.add_argument(
            '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
        parser.add_argument(
            '--alpha_upsampler',
            type=str,
            default='realesrgan',
            help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
        parser.add_argument(
            '--ext',
            type=str,
            default='auto',
            help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
        
        args, unknown = parser.parse_known_args()

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_path = os.path.join('./weights/', "RealESRGAN_x4plus" + '.pth')
        
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=args.fp32,
        )

        self.args = args
        self.upsampler = upsampler

    def gan_ext(self, img_cv2):
            print("[OK] Start the GAN enhancement.")
            img = img_cv2
            try:
                output, _ = self.upsampler.enhance(img, outscale=self.args.outscale)
                return output, True
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                return img, False


app = Flask(__name__, static_folder="./static/")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def predicts():

    if request.method == 'POST':
        if 'filename' not in request.files:
            print("[Error] The file is empty.")
            return redirect(request.url)
        
        file = request.files['filename']
        print("[OK] Get file.")

        if file and allowed_file(file.filename):

            img = Image.open(file)
            width, height = img.size
            if width > 800 or height > 800:
                print("[Error] The file size is too big!!")
                error_message = "Error: 图像尺寸超过了限制。"
                return render_template('index.html', error_message=error_message)

            buf = io.BytesIO()
            print("[OK] prepare buf for org image.")
            org_img = Image.open(file).convert('RGB')
            print("[OK] open org image.")
            org_img.save(buf, format='JPEG')
            print("[OK] save org image to buf.")
            org_image_data = buf.getvalue()
            base64_str_org = base64.b64encode(org_image_data).decode('utf-8')
            base64_data_org = 'data:image/jpeg;base64,{}'.format(base64_str_org)
            print("[OK] save to base64_str_org")

            img_array = np.array(bytearray(buf.getvalue()), dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            print("[OK] convert to img_cv (CV2 format).")
            
            GAN = esrgan(4)
            print("[OK] Initialize GAN.")
            output, ret = GAN.gan_ext(img_cv)

            if ret:
                print("[OK] Success the GAN.")
                up_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                print("[OK] convert to up_image (CV2 format).")
                buf = io.BytesIO()
                up_image.save(buf, format='JPEG')
                up_image_data = buf.getvalue()
                print("[OK] save up image to buf.")
                base64_str_up = base64.b64encode(up_image_data).decode('utf-8')
                base64_data_up = 'data:image/jpeg;base64,{}'.format(base64_str_up)
                print("[OK] save to base64_str_up")
                return render_template('result.html', image_org=base64_data_org, image_up=base64_data_up)

        error_message="Error: 文件格式只支持JPG或PNG格式" 
        return render_template("index.html", error_message=error_message)

    elif request.method == 'GET':
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)