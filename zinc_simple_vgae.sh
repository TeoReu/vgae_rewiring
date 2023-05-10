for model in "GCN"
do
  for transform in "no"
  do
    for layers in 4 6
    do
      for alpha in -10 -5 -1 0 0.01 0.1 1 5
      do
        python zinc_simple_vgae.py --model $model --layers $layers --file_name "peptides_simple_vgae_study_over_alpha" --transform $transform --alpha $alpha
      done
    done
  done
done