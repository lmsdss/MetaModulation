## miniImagenet-S
# 1-shot
python  main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5     --num_filters=32 --mix=1 --ratio=0.2
python  main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=15 --num_classes=5     --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch_start=500 --test_epoch_end=50500
# 5-shot
python  main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5     --num_filters=32 --mix=1 --ratio=0.2
python  main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=15 --num_classes=5     --num_filters=32 --mix=1 --ratio=0.2 --train=0 --test_epoch_start=500 --test_epoch_end=50500

