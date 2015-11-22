for j in 25 50 100 200
do
	for i in 0 1 2 3
	do
	python plt_result.py $j $i &
	done
	wait
done

