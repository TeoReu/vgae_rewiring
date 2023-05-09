for model in "PNA"
do
  for transform in "laplacian"
  do
    for layers in 6
    do
      for alpha in -20 -10 -5 -1 0 0.001 0.01 0.1 0.5 1 2
      do
        python zinc_simple_vgae.py --model $model --layers $layers --file_name "peptides_simple_vgae_study_over_alpha" --transform $transform --alpha $alpha
      done
    done
  done
done