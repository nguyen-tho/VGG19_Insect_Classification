from sklearn.model_selection import train_test_split


def split_data_3_parts(image, label, train_ratio, val_ratio, test_ratio):
    image_train_val, image_test, label_train_val, label_test = train_test_split(image, label, test_size=test_ratio,  stratify = label , random_state = 0)
    remaining_ratio = val_ratio / (train_ratio + val_ratio)
    image_train, image_val, label_train, label_val = train_test_split(image_train_val, label_train_val, test_size=remaining_ratio, stratify=label_train_val, random_state=0)

    return image_train, image_val, image_test, label_train, label_val, label_test

def split_data_2_parts(image, label, test_ratio):
    image_train, image_test, label_train, label_test = train_test_split(image, label, test_size=test_ratio,  stratify = label , random_state = 0 )
    return image_train, image_test, label_train, label_test






