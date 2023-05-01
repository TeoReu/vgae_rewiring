for model in "vr"
do
  for layers in 4
  do
    for vae_layers in 2 4
    do
      for alpha in 0.1
      do
          for threshold in 0.65
          do
            for transform in 0 1
            do
              for nr in 123 234 345
              do
                python zinc_classifier.py --model $model --vae_layers $vae_layers --threshold $threshold --layers $layers --alpha $alpha --file_name "results_regression_zinc" --transform $transform --nr $nr
              done
            done
          done
      done
    done
  done
done

for model in "simple"
do
  for layers in 4
  do
    for transform in 0 1
    do
      for nr in 123 234 345
      do
        python zinc_classifier.py --model $model --layers $layers --file_name "results_regression_zinc" --transform $transform --nr $nr
      done
    done
  done
done