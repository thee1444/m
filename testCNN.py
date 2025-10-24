import cv2
import numpy as np
from memryx import Engine
import time

# تحميل النموذج المحول إلى DFP
engine = Engine("my_model.dfp")

# إعداد الكاميرا (0 = الكاميرا الافتراضية)
cap = cv2.VideoCapture(0)

# حجم الإدخال كما درّبت عليه النموذج (عدّلها لو تدريبك مختلف)
IMG_SIZE = (224, 224)

# ثابت لتصفية النتائج
THRESHOLD = 0.6  # عدّل حسب أداء النموذج (مثلاً 0.5 أو 0.7)

print("✅ Running fall detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تجهيز الإطار للنموذج
    img_resized = cv2.resize(frame, IMG_SIZE)
    img_input = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    # تنفيذ الاستدلال على جهاز MemryX
    start = time.time()
    output = engine.run(img_input)
    end = time.time()

    # قراءة النتيجة (توقع سقوط أو لا)
    prob = float(output[0]) if np.ndim(output) == 1 else float(output[0][0])
    label = "FALL DETECTED!" if prob >= THRESHOLD else "Safe"

    # رسم النتائج على الشاشة
    color = (0, 0, 255) if label == "FALL DETECTED!" else (0, 255, 0)
    cv2.putText(frame, f"{label} ({prob:.2f})", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"FPS: {1/(end-start):.1f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Real-Time Fall Detection (MemryX)", frame)

    # خروج عند الضغط على Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
