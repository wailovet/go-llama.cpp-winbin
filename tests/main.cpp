#include <iostream>

int main()
{
    uint8_t a[2] = {136, 34};
    std::cout << float((a[0] & 0xf)) << std::endl;
    std::cout << float(a[0] >> 4) << std::endl;

    std::cout << float((a[1] & 0xf)) << std::endl;
    std::cout << float(a[1] >> 4) << std::endl;
    std::cout << "--" << std::endl;

    int *b = (int *)a; 
    std::cout << float((b[0] >> 0) & 0xf) << std::endl;
    std::cout << float((b[0] >> 4) & 0xf) << std::endl;
    std::cout << float((b[0] >> 8) & 0xf) << std::endl;
    std::cout << float((b[0] >> 12) & 0xf) << std::endl; 
}