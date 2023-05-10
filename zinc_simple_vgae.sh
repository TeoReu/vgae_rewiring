for dataset in "ZINC"
do
  for model in "PNA"
  do
    for transform in "laplacian"
    do
      for layers in 4 6
      do
        for alpha in   0 -1 0.01 -5 0.1 -10 1 5
        do
          python simple_vgae.py --model $model --layers $layers --file_name "peptides_simple_vgae_study_over_alpha" --transform $transform --alpha $alpha --dataset $dataset
        done
      done
    done
  done
done