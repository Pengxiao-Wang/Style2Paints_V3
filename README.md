# Style2Paints_V3
Reimplementation of Style2Paints V3 ([https://github.com/lllyasviel/style2paints/blob/master/papers/sa.pdf]())

## Result

<table>
    <tr>
        <td ><center><img src="./example/1_sketch.png" width="200px">figure 1_1  新垣结衣1 </center></td>
        <td ><center><img src="./example/1_color.png" width="200px">figure 1_2  新垣结衣1 </center></td>
    </tr>
    <tr>
    	<td ><center><img src="./example/2_sketch.png" width="200px">figure 2_1  新垣结衣2 </center></td>
    	<td ><center><img src="./example/2_color.png" width="200px">figure 2_2  新垣结衣2 </center></td>
	</tr>
    <tr>
    	<td ><center><img src="./example/3_sketch.png" width="200px">figure 3_1  新垣结衣3 </center></td>
    	<td ><center><img src="./example/3_color.png" width="200px">figure 3_2  新垣结衣3 </center></td>
	</tr>
    <tr>
    	<td ><center><img src="./example/4_sketch.png" width="200px">figure 4_1  新垣结衣4 </center></td>
    	<td ><center><img src="./example/4_color.png" width="200px">figure 4_2  新垣结衣4 </center></td>
	</tr>
    <tr>
    	<td ><center><img src="./example/5_sketch.png" width="200px">figure 5_1  新垣结衣5 </center></td>
    	<td ><center><img src="./example/5_color.png" width="200px">figure 5_2  新垣结衣5 </center></td>
	</tr>
    <tr>
    	<td ><center><img src="./example/6_sketch.png" width="200px">figure 6_1  新垣结衣6 </center></td>
    	<td ><center><img src="./example/6_color.png" width="200px">figure 6_2  新垣结衣6 </center></td>
	</tr>
    <tr>
    	<td ><center><img src="./example/7_sketch.png" width="200px">figure 7_1  新垣结衣7 </center></td>
    	<td ><center><img src="./example/7_color.png" width="200px">figure 7_2  新垣结衣7 </center></td>
	</tr>
    <tr>
    	<td ><center><img src="./example/8_sketch.png" width="200px">figure 8_1  新垣结衣8 </center></td>
    	<td ><center><img src="./example/8_color.png" width="200px">figure 8_2  新垣结衣8 </center></td>
	</tr>
    <tr>
    	<td ><center><img src="./example/9_sketch.png" width="200px">figure 9_1  新垣结衣9 </center></td>
    	<td ><center><img src="./example/9_color.png" width="200px">figure 9_2  新垣结衣9 </center></td>
	</tr>
</table>



## Step1 : Dataset Simulation

#### ​	One should modify simulate_step\*.ipynd or simulate_step\*.py with your own data path before runing this script. 

### Simulate_step1 : Random Region Proposal and Pasting

<center><img src="./example/Random Region Proposal and Pasting.png" width="400px"></center>

​	See script : simulate_step1.ipynb 

### Simulate_step2 : Random transform

<center>
    <img src="./example/Random transform.png" width="400px">
</center>

​	See script : simulate_step2.ipynb 

### Simulate_step3 : Random color spray

<center>
    <img src="./example/Random color spray.png" width="400px">
</center>

​	I merged this part with the Pytorch-data loader. Refer to ./Pytorch-Style2paints/dataset_multi.py

### Effect Picture

<table>
    <tr>
        <td ><center>Ground truth</center></td>
        <td ><center>Color draft</center></td>
    </tr>
    <tr>
    	<td ><center><img src="./example/gt_1.png" width="200px"></center></td>
    	<td ><center><img src="./example/df_1.png" width="200px"></center></td>
	</tr>
    <tr>
    	<td ><center><img src="./example/gt_2.png" width="200px"></center></td>
    	<td ><center><img src="./example/df_2.png" width="200px"></center></td>
	</tr>
</table>

