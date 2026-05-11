#include "Ver.h"

void setup()
{
    Serial.begin(115200);
    delay(15000);
	
	//...
	Serial.println("Setup done");
}

void loop()
{
	//...

    // Allow CPU to switch to other tasks.
    yield();
}

