import litdata as ld

train_dataset = ld.StreamingDataset("./fast_data", shuffle=True, drop_last=True)
train_dataloader = ld.StreamingDataLoader(train_dataset)

for sample in train_dataloader:
    img, cls = sample["image"], sample["class"]
    print(cls)
