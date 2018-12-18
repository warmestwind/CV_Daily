# https://realpython.com/python-exceptions/

for i in range(5):
    try:
        if i == 3:
            raise Exception('i should not be 3. The value of x was: {}'.format(i))
        else:
            print(i)
    except Exception as exc:
         #print exc
         #continue
        break
        #pass
        
