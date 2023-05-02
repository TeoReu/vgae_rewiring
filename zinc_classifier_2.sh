for model in "vr"
do
  for layers in 4
  do
    for vae_layers in 1 4
    do
      for alpha in 0.0 0.1 1.0
      do
          for threshold in 0.65 0.8
          do
            for transform in 0 1
            do
              for nr in 123
              do
                python zinc_classifier.py --model $model --vae_layers $vae_layers --threshold $threshold --layers $layers --alpha $alpha --file_name "results_regression_zinc" --transform $transform --nr $nr
              done
            done
          done
      done
    done
  done
done

