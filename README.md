# Python code for "Deep Learning for Massive MIMO CSI Feedback"
(c) 2018 Wang-Ting Shih and Chao-Kai Wen e-mail: sydney2317076@gmail.com and chaokai.wen@mail.nsysu.edu.tw

## Introdution
This repository contains the original models described in the paper "Deep Learning for Massive MIMO CSI Feedback" (https://ieeexplore.ieee.org/document/8322184/).

## Citation

If you use these models in your research, please cite:

	@article{He2015,
		author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
		title = {Deep Residual Learning for Image Recognition},
		journal = {arXiv preprint arXiv:1512.03385},
		year = {2015}
	}
## Environment
- Python 3.5(or 3.6)
- Keras
- TensorFlow as a backend of Keras (1.7.0)

## Steps to start

### Step1. Download the Model
There are two models in the paper:
- CsiNet: An antoencoder for reconstruct the CSI
- CS-CsiNet: Only learns to recover CSI from CS random linear measurements

We also provide two types of code:
- onlytest: Only to reproduce the results which provide in the paper. Also, we provide the model and weight which we have trained in the folder called 'saved_model'.
- train: Training by yourself

### Step2. Data Preparation
Download the data from https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing. After you got the data, put the data as shown below.
```
*.py
saved_model/
  *.h5
  *.json
data/
  *.mat
```

### Step3. Run the file
Now, you are ready to run any *.py to get the result.

## Result
The table shows same as the results in paper:

<table>
   <tr>
      <td></td>
      <td></td>
      <td>Indoor</td>
      <td></td>
      <td>Outdoor</td>
      <td></td>
   </tr>
   <tr>
      <td>gamma</td>
      <td>Methods</td>
      <td>NMSE</td>
      <td>rho</td>
      <td>NSME</td>
      <td>rho</td>
   </tr>
   <tr>
      <td>1/4</td>
      <td>LASSO</td>
      <td>-7.59</td>
      <td>0.91</td>
      <td>-5.08</td>
      <td>0.82</td>
   </tr>
   <tr>
      <td></td>
      <td>BM3D-AMP</td>
      <td>-4.33</td>
      <td>0.80</td>
      <td>-1.33</td>
      <td>0.52</td>
   </tr>
   <tr>
      <td></td>
      <td>TVAL3</td>
      <td>-14.87</td>
      <td>0.97</td>
      <td>-6.90</td>
      <td>0.88</td>
   </tr>
   <tr>
      <td></td>
      <td>CS-CsiNet</td>
      <td>-11.82</td>
      <td>0.96</td>
      <td>-6.69</td>
      <td>0.87</td>
   </tr>
   <tr>
      <td></td>
      <td>CsiNet</td>
      <td>-17.36</td>
      <td>0.99</td>
      <td>-8.75</td>
      <td>0.91</td>
   </tr>
   <tr>
      <td>1/16</td>
      <td>LASSO</td>
      <td>-2.72</td>
      <td>0.70</td>
      <td>-1.01</td>
      <td>0.46</td>
   </tr>
   <tr>
      <td></td>
      <td>BM3D-AMP</td>
      <td>0.26</td>
      <td>0.16</td>
      <td>0.55</td>
      <td>0.11</td>
   </tr>
   <tr>
      <td></td>
      <td>TVAL3</td>
      <td>-2.61</td>
      <td>0.66</td>
      <td>-0.43</td>
      <td>0.45</td>
   </tr>
   <tr>
      <td></td>
      <td>CS-CsiNet</td>
      <td>-6.09</td>
      <td>0.87</td>
      <td>-2.51</td>
      <td>0.66</td>
   </tr>
   <tr>
      <td></td>
      <td>CsiNet</td>
      <td>-8.65</td>
      <td>0.93</td>
      <td>-4.51</td>
      <td>0.79</td>
   </tr>
   <tr>
      <td>1/32</td>
      <td>LASSO</td>
      <td>-1.03</td>
      <td>0.48</td>
      <td>-0.24</td>
      <td>0.27</td>
   </tr>
   <tr>
      <td></td>
      <td>BM3D-AMP</td>
      <td>24.72</td>
      <td>0.04</td>
      <td>22.66</td>
      <td>0.04</td>
   </tr>
   <tr>
      <td></td>
      <td>TVAL3</td>
      <td>-0.27</td>
      <td>0.33</td>
      <td>0.46</td>
      <td>0.28</td>
   </tr>
   <tr>
      <td></td>
      <td>CS-CsiNet</td>
      <td>-4.67</td>
      <td>0.83</td>
      <td>-0.52</td>
      <td>0.37</td>
   </tr>
   <tr>
      <td></td>
      <td>CsiNet</td>
      <td>-6.24</td>
      <td>0.89</td>
      <td>-2.81</td>
      <td>0.37</td>
   </tr>
   <tr>
      <td>1/64</td>
      <td>LASSO</td>
      <td>-0.14</td>
      <td>0.22</td>
      <td>-0.06</td>
      <td>0.12</td>
   </tr>
   <tr>
      <td></td>
      <td>BM3D-AMP</td>
      <td>0.22</td>
      <td>0.04</td>
      <td>25.45</td>
      <td>0.03</td>
   </tr>
   <tr>
      <td></td>
      <td>TVAL3</td>
      <td>0.63</td>
      <td>0.11</td>
      <td>0.76</td>
      <td>0.19</td>
   </tr>
   <tr>
      <td></td>
      <td>CS-CsiNet</td>
      <td>-2.46</td>
      <td>0.68</td>
      <td>-0.22</td>
      <td>0.28</td>
   </tr>
   <tr>
      <td></td>
      <td>CsiNet</td>
      <td>-5.84</td>
      <td>0.87</td>
      <td>-1.93</td>
      <td>0.59</td>
   </tr>
</table>
