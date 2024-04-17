import re
import numpy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import skimage as ski
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', output_hidden_states=True)
vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1')
label_to_id = {'fake': 0, 'real': 1}

def load_data(dir):
    texts = []
    images = []
    rumor = []
    with open(dir + "test_nonrumor.txt", 'r', encoding="utf8") as file:
        c = 0
        for line in file:
            if c == 0:
                c = (c+1) % 3
                continue
            if c == 1:
                images.append(line)
                c = (c + 1) % 3
                continue
            if c == 2:
                texts.append(line)
                rumor.append(1)
                c = (c + 1) % 3
                continue
    with open(dir + "test_rumor.txt", 'r', encoding="utf8") as file:
        c = 0
        for line in file:
            if c == 0:
                c = (c+1) % 3
                continue
            if c == 1:
                images.append(line)
                c = (c + 1) % 3
                continue
            if c == 2:
                texts.append(line)
                rumor.append(0)
                c = (c + 1) % 3
                continue

    X_text = []
    X_image = []
    labels = []
    for i in range(len(images)):
        url = images[i].split('|')[0]
        try:
            img = ski.io.imread(url)
            if re.search("gif$", url):
                img = img[0]
            img = ski.transform.resize(img, (244, 244, 3), order=3)
            img = img.transpose(2, 0, 1)
        except:
            print(url)
            print(img.shape)
            continue
        X_text.append(texts[i])
        X_image.append(img)
        labels.append(rumor[i])

    return X_text, X_image, labels

def textual_feature_extractor(X_text):

    X_text_encodings = []

    for text in X_text:

        indexed_tokens = tokenizer.encode(text)
        l = len(indexed_tokens)
        if l > 200:
            sep_token = indexed_tokens[-1]
            indexed_tokens = indexed_tokens[:199]
            indexed_tokens.append(sep_token)

        segments = [1] * len(indexed_tokens)

        while len(indexed_tokens) < 200:
            indexed_tokens.append(0)
            segments.append(0)

        segments_tensors = torch.tensor([segments])
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            encoded_layers = bert(tokens_tensor, segments_tensors)
        X_text_encodings.append(encoded_layers.hidden_states[-1][0][0])

    return torch.stack(X_text_encodings)


def visual_feature_extractor(X_images):
    activation = {}

    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook
    image_encodings = []
    for image in X_images:
        image_tensor = torch.tensor(image.astype(numpy.float32, casting="same_kind"))
        image_tensor = image_tensor.unsqueeze(0)
        layer = getattr(vgg19.classifier, '5')

        with torch.no_grad():
            layer.register_forward_hook(getActivation("secondLast"))
            outputs = vgg19(image_tensor)
            image_encodings.append(outputs)
    return torch.stack(image_encodings)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.reluText1 = nn.ReLU()
        self.fcText1 = nn.Linear(768, 768)
        self.textDropout1 = nn.Dropout(p=0.4)
        self.reluText2 = nn.ReLU()
        self.fcText2 = nn.Linear(768, 32)
        self.textDropout2 = nn.Dropout(p=0.4)
        self.reluImage1 = nn.ReLU()
        self.fcImage1 = nn.Linear(1000, 32)
        self.imgDropout1 = nn.Dropout(p=0.4)
        self.reluOut1 = nn.ReLU()
        self.fcOut1 = nn.Linear(64, 35)
        self.outDropout = nn.Dropout(p=0.4)
        self.reluOut2 = nn.ReLU()
        self.fcOut2 = nn.Linear(35, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, img):

        text = self.reluText1(text)
        text = self.fcText1(text)
        text = self.textDropout1(text)
        text = self.reluText2(text)
        text = self.fcText2(text)
        #text = self.textDropout2(text)

        img = self.reluImage1(img)
        img = self.fcImage1(img)
        img = self.imgDropout1(img)

        img = torch.squeeze(img, 1)
        concat = torch.cat((text, img), 1)

        out = self.reluOut1(concat)
        out = self.fcOut1(out)
        out = self.outDropout(out)
        out = self.reluOut2(out)
        out = self.fcOut2(out)
        out = self.sigmoid(out)

        return out

"""
X_text, X_images, labels= load_data("./data/weibo/")
np.save("./data/weibo/X_text", X_text)
np.save("./data/weibo/X_images", X_images)
np.save("./data/weibo/labels", labels)
"""

X_text = np.load("./data/weibo/X_text.npy")
X_images = np.load("./data/weibo/X_images.npy")
labels = np.load("./data/weibo/labels.npy")

index = np.random.permutation(len(labels))
X_text = X_text[index]
X_images = X_images[index]
labels = labels[index]

labels = torch.tensor(labels, dtype=torch.float32)
labels = torch.unsqueeze(labels, 1)

X_text_encodings = textual_feature_extractor(X_text)
X_image_encodings = visual_feature_extractor(X_images)

"""
X_text_test, X_images_test, labels_test = load_data("./data/weibo/")
np.save("./data/weibo/X_text_test", X_text)
np.save("./data/weibo/X_images_test", X_images)
np.save("./data/weibo/labels_test", labels)
"""

X_text_test = np.load("./data/weibo/X_text_test.npy")
X_images_test = np.load("./data/weibo/X_images_test.npy")
labels_test = np.load("./data/weibo/labels_test.npy")

labels_test = torch.tensor(labels_test, dtype=torch.float32)
labels_test = torch.unsqueeze(labels_test, 1)

X_text_encodings_test = textual_feature_extractor(X_text_test)
X_image_encodings_test = visual_feature_extractor(X_images_test)

clf = Classifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

best_val_acc = 0.0
patience = 1
c = 0
print("Starting training")
for epoch in range(10):
    running_loss = 0.0
    for i in range(0, len(labels), 256):

        optimizer.zero_grad()

        outputs = clf(X_text_encodings[i:i+256], X_image_encodings[i:i+256])
        loss = criterion(outputs, labels[i:i+256])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 256:.3f}')
        running_loss = 0.0

    clf.eval()
    with torch.no_grad():
        val_outputs = clf(X_text_encodings_test, X_image_encodings_test)
        predictions = torch.round(val_outputs)
        val_acc = accuracy_score(predictions, labels_test)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(clf.state_dict(), "./data/weibo/model")
        c = 0
    else:
        c += 1
        if c >= patience:
            print("Early stop, best validation accuracy: " + str(best_val_acc))
            break


"""Test model"""


clf = Classifier()
clf.load_state_dict(torch.load("./data/weibo/model"))
with torch.no_grad():
    clf.eval()
    outputs = clf(X_text_encodings_test, X_image_encodings_test)
    predictions = torch.round(outputs)
"""Evaluate"""
acc = accuracy_score(predictions.detach().numpy(), labels_test)
recall = recall_score(predictions.detach().numpy(), labels_test)
precision = precision_score (predictions.detach().numpy(), labels_test)
f1 = f1_score(predictions.detach().numpy(), labels_test)

print('Accuracy: ' + str(acc))
print('Recall: ' + str(recall))
print('Precision ' + str(precision))
print('F1 score ' + str(f1))