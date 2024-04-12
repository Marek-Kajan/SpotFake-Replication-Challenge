import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import skimage as ski
import os
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#pretrained models
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', output_hidden_states=True)
vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1')

label_to_id = {'fake': 0, 'real': 1}

def load_train_data(dir, image_dir):
    data = pd.read_csv(dir + "train_posts.txt", sep='\t')
    data = shuffle(data)
    img_data = pd.read_csv(dir + "tweets_images.txt", sep='\t')
    files = pd.DataFrame(os.listdir(image_dir))
    files = files[0].str.split('.', expand=True)
    files.columns = ['name', 'ext']
    X_text = []
    X_image = []
    labels = []
    c = 0
    for post_id in data['post_id']:
        image_ids = img_data[img_data['tweet_id'] == post_id]['image_id(s)']
        for images in image_ids:
            for image in images.split(','):
                try:
                    ext = files[files['name'] == image]['ext']
                    if ext.empty:
                        raise FileNotFoundError
                    extension = ext.values[0]
                    img = ski.io.imread(image_dir + "/" + image + "." + extension)
                    if extension == 'gif':
                        #take the 1st frame of gif
                        img = img[0]
                    img = ski.transform.resize(img, (244, 244, 3), order=3)
                    img = img.transpose(2, 0, 1)
                    X_text.append(data[data['post_id'] == post_id]['post_text'])
                    X_image.append(img)
                    labels.append(label_to_id[data[data['post_id'] == post_id]['label'].iloc[0]])
                except (FileNotFoundError, OSError):
                    print(image)
                    c += 1
                    continue
    print(c)
    return X_text, X_image, labels

def load_test_dataset(dir, image_dir):
    data = pd.read_csv(dir + "test_posts.txt", sep='\t')
    data = shuffle(data)
    files = pd.DataFrame(os.listdir(image_dir))
    files = files[0].str.split('.', expand=True)
    files.columns = ['name', 'ext']
    X_text = []
    X_image = []
    labels = []
    c = 0

    for id in data.index:
        for image in data['image_id'][id].split(','):
            try:
                ext = files[files['name'] == image]['ext']
                if ext.empty:
                    raise FileNotFoundError
                extension = ext.values[0]
                if extension == 'txt':
                    raise OSError
                img = ski.io.imread(image_dir + "/" + image + "." + extension)
                if extension == 'gif':
                    #take the 1st frame of gif
                    img = img[0]
                img = ski.transform.resize(img, (244, 244, 3), order=3)
                img = img.transpose(2, 0, 1)
                X_text.append(data['post_text'][id])
                X_image.append(img)
                labels.append(label_to_id[data['label'][id]])
            except (FileNotFoundError, OSError):
                c += 1
                continue
    print(c)
    return X_text, X_image, labels

def textual_feature_extractor(X_text):

    X_text_encodings = []

    for text in X_text:

        indexed_tokens = tokenizer.encode(text[0])
        l = len(indexed_tokens)
        if l > 23:
            sep_token = indexed_tokens[-1]
            indexed_tokens = indexed_tokens[:22]
            indexed_tokens.append(sep_token)

        segments = [1] * len(indexed_tokens)

        while len(indexed_tokens) < 23:
            indexed_tokens.append(0)
            segments.append(0)

        segments_tensors = torch.tensor([segments])
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            encoded_layers = bert(tokens_tensor, segments_tensors)
        X_text_encodings.append(encoded_layers.hidden_states[-2][0][0])

    return torch.stack(X_text_encodings)


def visual_feature_extractor(X_images):
    activation = {}

    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook
    image_encodings = []
    for image in X_images:
        ##image = image.transpose(2, 0, 1)
        image_tensor = torch.tensor(image.astype(numpy.float32, casting="same_kind"))
        image_tensor = image_tensor.unsqueeze(0)
        layer = getattr(vgg19.classifier, '5')

        with torch.no_grad():
            layer.register_forward_hook(getActivation("secondLast"))
            _ = vgg19(image_tensor)
            image_encodings.append(activation["secondLast"])
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
        self.fcImage1 = nn.Linear(4096, 2742)
        self.imgDropout1 = nn.Dropout(p=0.4)
        self.reluImage2 = nn.ReLU()
        self.fcImage2 = nn.Linear(2742, 32)
        self.imgDropout2 = nn.Dropout(p=0.4)
        self.concatDropout = nn.Dropout(p=0.4)
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
        img = self.reluImage2(img)
        img = self.fcImage2(img)
        #img = self.imgDropout2(img)

        img = torch.squeeze(img, 1)
        concat = torch.cat((text, img), 1)
        concat = self.concatDropout(concat)

        out = self.reluOut1(concat)
        out = self.fcOut1(out)
        out = self.outDropout(out)
        out = self.reluOut2(out)
        out = self.fcOut2(out)
        out = self.sigmoid(out)

        return out
"""
X_text, X_images, labels= load_data("./data/twitter/", "./data/twitter/Mediaeval2015_DevSet_Images")
np.save("./data/twitter/X_text", X_text)
np.save("./data/twitter/X_images", X_images)
np.save("./data/twitter/labels", labels)
"""
X_text = np.load("./data/twitter/X_text.npy", allow_pickle=True)
X_images = np.load("./data/twitter/X_images.npy")
labels = np.load("./data/twitter/labels.npy")

#randomize dataset
index = np.random.permutation(len(labels))
X_text = X_text[index]
X_images = X_images[index]
labels = labels[index]

labels = torch.tensor(labels, dtype=torch.float32)
labels = torch.unsqueeze(labels, 1)

"""
X_text, X_images, labels = load_test_dataset("./data/twitter/", "./data/twitter/Mediaeval2016_TestSet_Images")
np.save("./data/twitter/X_text_test", X_text)
np.save("./data/twitter/X_images_test", X_images)
np.save("./data/twitter/labels_test", labels)
"""

X_text_test = np.load("./data/twitter/X_text_test.npy", allow_pickle=True)
X_images_test = np.load("./data/twitter/X_images_test.npy")
labels_test = np.load("./data/twitter/labels_test.npy")

labels_test = torch.tensor(labels_test, dtype=torch.float32)
labels_test = torch.unsqueeze(labels_test, 1)

X_text_encodings_test = textual_feature_extractor(X_text_test)
X_image_encodings_test = visual_feature_extractor(X_images_test)


X_text_encodings = textual_feature_extractor(X_text)
X_image_encodings = visual_feature_extractor(X_images)

clf = Classifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=0.0005)

best_val_acc = 0.0
patience = 3
c = 0
print("Starting training")
for epoch in range(10):
    running_loss = 0.0
    for i in range(0, len(labels), 256):
        clf.train()
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
        torch.save(clf.state_dict(), "./data/twitter/model")
        c = 0
    else:
        c += 1
        if c >= patience:
            print("Early stop, best validation accuracy: " + str(best_val_acc))
            break


"""Test model"""

clf = Classifier()
clf.load_state_dict(torch.load("./data/twitter/model"))

with torch.no_grad():
    clf.eval()
    outputs = clf(X_text_encodings_test, X_image_encodings_test)
    predictions = torch.round(outputs)

"""Evaluate model"""
acc = accuracy_score(predictions.detach().numpy(), labels_test)
recall = recall_score(predictions.detach().numpy(), labels_test)
precision = precision_score(predictions.detach().numpy(), labels_test)
f1 = f1_score(predictions.detach().numpy(), labels_test)

print('Accuracy: ' + str(acc))
print('Recall: ' + str(recall))
print('Precision ' + str(precision))
print('F1 score ' + str(f1))
