const axios = require("axios");

const login = async (email, password) => {
  try {
    const response = await axios.post(
      "http://localhost:8000/token",
      new URLSearchParams({
        username: email, // Using email here
        password: password,
      }),
      {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      }
    );

    console.log(response.data);
  } catch (error) {
    console.error(
      "Error logging in:",
      error.response ? error.response.data : error.message
    );
  }
};

// Example usage - now using email for login
login("maidang24112004@gmail.com", "haidang2004");
