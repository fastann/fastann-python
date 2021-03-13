# FastANN Python Binding


## Install
thanks for [setuptools-rust](https://github.com/PyO3/setuptools-rust)

### prerequirement

```
pip install -r setuptools-rust setuptools wheel
```

### release
```
python3 setup.py install --user
```

### debug
```
python3 setup.py develop --user
```

## Example

see `example.py` file

# Develop

## mac 
```
cd fastann
cargo rustc --release -- -C link-arg=-undefined -C link-arg=dynamic_lookup
cp -R target/release/libfastann.dylib target/release/fastann.so
```

## windows 
```
cd fastann
cargo rustc --release
cp -R target/release/libfastann.dll target/release/fastann.pyd
```