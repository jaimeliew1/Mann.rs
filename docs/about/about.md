## **About the author**
Hi, I'm Jaime. I wrote Mann.rs during my PhD on dynamic wind farm modelling. At the time, I needed to generate extremely large, high-resolution turbulence boxes—but no existing software could handle that scale efficiently. So I revisited the Mann turbulence model and implemented several numerical optimizations which significantly improved the memory and computational efficiency of the model.

Mann.rs is the result of that work. It’s designed to be fast, scalable, and easy to integrate into simulation workflows. I’m happy to share it with the world as open source code, and I hope it proves useful to others working in wind energy, atmospheric modeling, or anywhere where synthetic turbulence is needed.

## **Citation**

The Mann.rs repository can be cited directly here

    Jaime Liew. (2022). jaimeliew1/Mann.rs: Publish Mann.rs v1.0.0 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.7254149

The numerical innovations in Mann.rs are described in:

    Liew, J., Riva, R., & Göçmen, T. (2023). Efficient Mann turbulence generation for offshore wind farms with applications in fatigue load surrogate modelling. Journal of Physics: Conference Series, 2626, 012050. DOI: 10.1088/1742-6596/2626/1/012050

The underlying Mann turbulence model is originally described in:

    Mann, J. (1998). Wind field simulation. Probabilistic Engineering Mechanics, 13(4), 269-282. DOI: 10.1016/S0266-8920(97)00036-2