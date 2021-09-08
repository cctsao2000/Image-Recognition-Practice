import turicreate as tc
import cv2
img_folder = "image"
data = tc.image_analysis.load_images(img_folder,with_path=True)

# label data and save
data['label'] = data['path'].apply(lambda path: 'pp' if 'pp' in path else 'gshock')
data.save("watch.sframe")

# split datasets and train model
train_data, test_data = data.random_split(0.8, seed=2)
model = tc.image_classifier.create(train_data, target='label')

# 測驗集準度測試
# predictions = model.predict(test_data)
# metrics = model.evaluate(test_data)
# print(metrics['accuracy'])

# predict image from internet by url
# img = tc.Image("https://www.luxurywatcher.com/uploads/article/2/6/26752/1557916941185.jpg")
# result = model.predict(img)
# print(result)

#predict image from webcam
cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Error: could not access webcam")
while True:
    ret, frame = cap.read()
    cv2.imshow("preview",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite("cv_webcamcapture.png", frame)
        break
cap.release()
cv2.destroyAllWindows()

img = tc.Image("cv_webcamcapture.png")
result = model.predict(img)
print(result)
