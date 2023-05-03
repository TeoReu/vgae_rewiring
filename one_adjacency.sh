for model in "vr"
do
  for layers in 1 2 4
  do
    for vae_layers in 1
    do
      for alpha in 0.0 0.1 1.0
      do
          for threshold in 0.2
          do
            for transform in 0 1
            do
              for nr in 234 123
              do
                python one_adjacency_zinc.py --model $model --vae_layers $vae_layers --threshold $threshold --layers $layers --alpha $alpha --file_name "one_adj" --transform $transform --nr $nr
              done
            done
          done
      done
    done
  done
done