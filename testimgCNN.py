from memryx import Engine
import cv2
import numpy as np

# تحميل النموذج على الجهاز
engine = Engine("my_model.dfp")

# قراءة صورة اختبار
img = cv2.imread("fall_sample.jpg")
img_resized = cv2.resize(img, (224, 224))
img_input = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

# تنفيذ الاستدلال
output = engine.run(img_input)

# عرض النتيجة
print("Model output:", output)
