for model in "vr"
do
  for layers in 4
  do
    for vae_layers in 2 4
    do
      for alpha in 0.1
      do
          for threshold in 0.65 0.75 0.9 0.95
          do
            for transform in 0 1
            do
              for nr in 123 234 345
              do
                python zinc_classifier.py --model $model --vae_layers $vae_layers --threshold $threshold --layers $layers --alpha $alpha --file_name "special_results_regression_zinc" --transform $transform --nr $nr
              done
            done
          done
      done
    done
  done
done