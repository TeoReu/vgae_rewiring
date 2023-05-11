for dataset in "ZINC"
do
  for model in "PNA"
  do
    for transform in "no"
    do
      for layers in 4
      do
        for alpha in  -10 -5 -1 0 0.1 0.5 1 2 3
        do
          python train_vgae.py --model $model --layers $layers --file_name "vgae_hl_124" --transform $transform --alpha $alpha --dataset $dataset
        done
      done
    done
  done
done