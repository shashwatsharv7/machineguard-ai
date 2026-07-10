# MachineGuard AI

This repo is my predictive maintenance project on the AI4I 2020 dataset.

I used the machine sensor values to predict `Machine failure`. The dataset is quite imbalanced, so I did not focus only on accuracy. I checked precision, recall and F1 also.

The input columns I used are:

- Type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

I avoided the failure-type columns because they already give away information about the target.

I tried a few models first, then kept Random Forest as the final model. I also added two simple features:

```text
temperature_difference = process temperature - air temperature
power_proxy = torque * rotational speed
```

The final model uses a threshold of `0.3` instead of `0.5`, because missing failures is worse than having a few extra false alarms.

Final test result:

```text
Accuracy:  0.988
Precision: 0.855
Recall:    0.779
F1 score:  0.815
```

To train and test locally:

```bash
pip install -r requirements.txt
python scripts/train_model.py
python scripts/predict_one.py
```

To run the API:

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Docker also works:

```bash
docker build -t machineguard-ai .
docker run -p 8000:8000 machineguard-ai
```

This is still a learning project. The dataset is synthetic, so I would not call this production-ready.
