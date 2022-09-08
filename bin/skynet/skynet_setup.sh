# Make sure to run this file at the top level directory of your project

rm data
rm ckpts
rm modules/bert-base-uncased
rm weights
ln -s /coc/dataset/sthvl data
ln -s /coc/pskynet3/sthvl/ckpts ckpts
ln -s /coc/pskynet3/sthvl/modules/bert-base-uncased modules/bert-base-uncased
ln -s /coc/pskynet3/sthvl/weights weights
