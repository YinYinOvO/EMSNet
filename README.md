## Notes
3.25 We have updated the training code and released the result graph and weights, while also correcting some errors in the original code.
## Datasets
All datasets are available in public.
* Download the DUTS-TR and DUTS-TE from [Here](http://saliencydetection.net/duts/#org3aad434)
* Download the DUT-OMRON from [Here](http://saliencydetection.net/dut-omron/#org96c3bab)
* Download the HKU-IS from [Here](https://sites.google.com/site/ligb86/hkuis)
* Download the ECSSD from [Here](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
* Download the PASCAL-S from [Here](http://cbs.ic.gatech.edu/salobj/)

## Data structure
<pre><code>
├── data
│   ├── DUTS-TE
│   │   ├── Train
│   │   │   ├── images
│   │   │   ├── masks
│   │   │   ├── edges
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
│   ├── DUT-OMRON
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
│   ├── HKU-IS
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
│   ├── ECSSD
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
│   ├── PASCAL-S
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
</code></pre>

## Test Code
<pre><code>
cd EMSNet
python main.py test --save_map True
</code></pre>

The result graph and corresponding weight file can be obtained in [Here]([http://cbs.ic.gatech.edu/salobj/](https://pan.baidu.com/s/1ZgQWjS4mZIzoHHzMZyKRog?pwd=kog9 )https://pan.baidu.com/s/1ZgQWjS4mZIzoHHzMZyKRog?pwd=kog9 ) (kog9)
