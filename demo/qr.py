import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

from DQM.utils.data import parent

import qrcode
import qrcode.image.svg
from PIL import Image


def make_qr(qr_data,qr_name,qr_fill='black',qr_back='white',logo=False,logo_path=None,logo_size=120,ext='png'):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=1,
    )

    qr.add_data(qr_data)
    qr.make(fit=True)
    if ext == 'svg':
        img = qr.make_image(fill_color=qr_fill, back_color=qr_back,image_factory=qrcode.image.svg.SvgImage)

    else:
        img = qr.make_image(fill_color=qr_fill, back_color=qr_back).convert('RGB')
    # img = qr.make_image(fill_color=qr_fill, back_color=qr_back,image_factory=qrcode.image.svg.SvgPathImage)

        if logo:
            if isinstance(logo_path,type(None)):
                raise RuntimeError('No has introducido una dirección para la imagen del logo.')

            logo_display = Image.open(f'{logo_path}')
            logo_display.thumbnail((logo_size, logo_size))
            logo_pos = ((img.size[0] - logo_display.size[0]) // 2, (img.size[1] - logo_display.size[1]) // 2)
            img.paste(logo_display, logo_pos)

    img.save(f'{parent}/misc/{qr_name}.{ext}')


qr_kwargs = {
    'qr_data' : 'https://github.com/magnarex/uo269426-tfg',
    'qr_fill' : 'black',
    'qr_back' : 'white',
    'logo' : True,
    'logo_path' : f'{parent}/misc/uniovi.jpg',
    'qr_name' : f'GitHub',
    'ext' : 'png',
    'logo_size' : 150
}
make_qr(**qr_kwargs)