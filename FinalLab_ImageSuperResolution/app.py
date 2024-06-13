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
import time
import torch
from thop import profile, clever_format

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

        print("[OK] 参数解析完成！")

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        print("[OK] 网络架构创建完成！")

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

        print("[OK] 模型已加载，初始化upsampler完成！")

        # Calculate model complexity (FLOPs and parameters)
        dummy_input = torch.randn(1, 3, 256, 256)
        flops, params = profile(model, inputs=(dummy_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"[Model Cost] FLOPs: {flops}, Params: {params}")

    def gan_ext(self, img_cv2):
            print("[OK] 已启动GAN增强")
            img = img_cv2
            try:
                start_time = time.time()
                output, _ = self.upsampler.enhance(img, outscale=self.args.outscale)
                end_time = time.time()
                print(f"[Time] GAN增强一共花费了 {end_time - start_time:.4f} 秒")
                return output, True
            except RuntimeError as error:
                print('Error', error)
                print('如果您遇到CUDA内存不足，请尝试减小`-tile`')
                return img, False


app = Flask(__name__, static_folder="./static/")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def predicts():

    if request.method == 'POST':
        print("[INFO] 已收到POST请求！")
        if 'filename' not in request.files:
            print("[Error] 文件为空！")
            return redirect(request.url)
        
        file = request.files['filename']
        print("[OK] 文件已获取！")

        if file and allowed_file(file.filename):
            print("[OK] 文件格式正确！")

            img = Image.open(file)
            width, height = img.size
            if width > 800 or height > 800:
                print("[Error] 图像尺寸超过了限制！")
                error_message = "Error: 图像尺寸超过了限制。"
                return render_template('index.html', error_message=error_message)

            buf = io.BytesIO()
            print("[OK] 准备缓冲区存储原始图像。")
            org_img = Image.open(file).convert('RGB')
            print("[OK] 原始图像已打开，转换为RGB格式。")
            org_img.save(buf, format='JPEG')
            print("[OK] 原始图像已保存到缓冲区。")
            org_image_data = buf.getvalue()
            base64_str_org = base64.b64encode(org_image_data).decode('utf-8')
            base64_data_org = 'data:image/jpeg;base64,{}'.format(base64_str_org)
            print("[OK] 原始图像已转换为base64格式。")

            img_array = np.array(bytearray(buf.getvalue()), dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            print("[OK] 原始图像已转换为CV2格式。")

            GAN = esrgan(4)
            print("[OK] ESRGAN模型已加载！")
            output, ret = GAN.gan_ext(img_cv)

            if ret:
                print("[OK] GAN增强完成！")
                up_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                print("[OK] 增强图像已转换为CV2格式。")
                buf = io.BytesIO()
                up_image.save(buf, format='JPEG')
                up_image_data = buf.getvalue()
                print("[OK] 增强图像已保存到缓冲区。")
                base64_str_up = base64.b64encode(up_image_data).decode('utf-8')
                base64_data_up = 'data:image/jpeg;base64,{}'.format(base64_str_up)
                print("[OK] 增强图像已转换为base64格式。")
                return render_template('result.html', image_org=base64_data_org, image_up=base64_data_up)

        error_message = "Error: 文件格式只支持JPG或PNG格式" 
        print("[Error] 文件格式错误！")
        return render_template("index.html", error_message=error_message)

    elif request.method == 'GET':
        print("[INFO] 已收到GET请求！")
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)