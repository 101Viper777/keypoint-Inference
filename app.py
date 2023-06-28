from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
from keras.models import load_model
import numpy as np
from PIL import Image as PilImage
from skimage.transform import resize

model = load_model('new_Mae.h5')

def crop_image(image, new_shape):
    current_shape = image.shape
    y_start = (current_shape[0] - new_shape[0]) // 2
    y_end = y_start + new_shape[0]
    x_start = (current_shape[1] - new_shape[1]) // 2
    x_end = x_start + new_shape[1]
    cropped_image = image[y_start:y_end, x_start:x_end, :]
    return cropped_image

class KivyCamera(KivyImage):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()

        if ret:
            img = PilImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            new_height = 1280
            aspect_ratio = img.size[0] / img.size[1]
            new_width = round(new_height * aspect_ratio)
            img = img.resize((new_width, new_height))

            resized_frame = np.array(img)

            cropped_frame = crop_image(resized_frame, (1280, 720, 3))
            cropped_np = np.array(cropped_frame)
            resized_pre_image = resize(cropped_np, (160, 90, 3), mode='constant')
            image_batch = np.expand_dims(resized_pre_image, axis=0)

            predictions = model.predict(image_batch)

            predicted_keypoints = predictions.reshape((4, 2))

            predicted_keypoints = predicted_keypoints.astype(int)

            # Rotate the keypoints 180 degrees
            max_y = cropped_frame.shape[0]
            max_x = cropped_frame.shape[1]
            predicted_keypoints = [(max_x - x, max_y - y) for (x, y) in predicted_keypoints]

            Rcropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)

            cropped_frame_rgb = cv2.rotate(Rcropped_frame_rgb, cv2.ROTATE_180)

            # Now mirror the image
            mirrored_frame = cv2.flip(cropped_frame_rgb, 1)

            # Mirror the keypoints
            mirrored_keypoints = [(max_x - x, y) for (x, y) in predicted_keypoints]

            # Copy the frame so you don't modify the original
            drawn_frame = mirrored_frame.copy()

            # Draw circles at the mirrored keypoint locations
            for x, y in mirrored_keypoints:
                cv2.circle(drawn_frame, (x, y), 3, (0, 255, 0), -1)  # Draw green circles of radius 3

            texture = Texture.create(size=(drawn_frame.shape[1], drawn_frame.shape[0]), colorfmt='bgr')

            texture.blit_buffer(drawn_frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')

            self.texture = texture

class MyApp(App):
    def build(self):
        layout = BoxLayout()
        self.capture = cv2.VideoCapture(2)
        self.my_camera = KivyCamera(capture=self.capture, fps=60)
        layout.add_widget(self.my_camera)
        return layout

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    MyApp().run()

