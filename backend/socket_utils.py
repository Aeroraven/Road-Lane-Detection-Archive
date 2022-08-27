import base64

import cv2


def base64_to_image(data,sid):
    img_data = data.split(',')[1]
    img_data = base64.b64decode(img_data)
    with open('C:\\WR\\'+str(sid)+'.png', 'wb') as f:
        f.write(img_data)
    return r'C:\\WR\\'+str(sid)+'.png'


def img_to_base64(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    encode_image = cv2.imencode(".jpg", img_array)[1]
    byte_data = encode_image.tobytes()
    base64_str = base64.b64encode(byte_data).decode("ascii")
    return base64_str
