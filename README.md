# face_classification

Face classification by PyTorch

<img src="https://raw.githubusercontent.com/ijiwarunahello/face_classification/doc/image/detection_sample.JPG" width=500>

## Environment

### OS

```sh
❯ sw_vers
ProductName:    Mac OS X
ProductVersion: 10.15.1
BuildVersion:   19B2106
```

### Python

```sh
❯ python3 -V
Python 3.7.5
```

## Library install

```sh
pip3 install -r requirements.txt
```

## How to use

| num | description | command | output |
| :--- | :--- | :--- | :--- |
| 1 | Learning image collection | `python3 image_collector.py -t <PERSON-NAME> - n <NUMBER>` | `./data` |
| 2 | Extract face | `python3 cut_face.py` | `./face_data` |
| 3 | Data augmentation | `python3 make_test_train.py` | `./test_data` `./train_data` |
| 4 | Model creation and learning | `python3 main.py --e <EPOCH_NUM> --save-model` | `.pt` |
| 5 | Detection | `python3 detect_perfume.py` | `./her_name_is` |

_Note: Save the images you want to classify in the `who_is_member` folder in advance_

## Special thanks

[PyTorchを使って日向坂46の顔分類をしよう！](https://qiita.com/coper/items/b1fd51062642d624e26f#6-%E6%96%B0%E8%A6%8F%E7%94%BB%E5%83%8F%E3%82%92%E7%94%A8%E3%81%84%E3%81%A6%E5%AD%A6%E7%BF%92%E6%B8%88%E3%81%BF%E3%83%A2%E3%83%87%E3%83%AB%E3%81%8B%E3%82%89%E5%90%8D%E5%89%8D%E3%82%92%E7%89%B9%E5%AE%9A%E3%81%97%E6%96%B0%E8%A6%8F%E7%94%BB%E5%83%8F%E3%81%AB%E6%8F%8F%E7%94%BB)
