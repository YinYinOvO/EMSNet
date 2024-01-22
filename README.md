## Notes
Due to file upload size limitations, we provide the source code for our MobileNet v2 version along with a pre-trained model. Specific data set information and test code refer below.
Our pre-trained model is in the results folder, and the MobileNet code and initial weights are sourced from EDN's open source code in the model and premodels folders, respectively.
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
cd SDNet_MobileNetv2
python main.py test --save_map True
</code></pre>