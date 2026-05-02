// Extracted from Project_Dzukku.xlsx — Master_Menu sheet
export const MENU_ITEMS = [
  // Veg Items
  { id: 'V001', category: 'Veg', name: 'Paneer Butter Masala', description: 'Cottage cheese cubes cooked in creamy tomato gravy', price: 180, status: 'Available', isSpecial: false, stock: 1, specialPrice: null, prepTime: '15 mins', emoji: '🧀' },
  { id: 'V002', category: 'Veg', name: 'Vegetable Biryani', description: 'Fragrant rice with mixed vegetables and aromatic spices', price: 150, status: 'Available', isSpecial: true, stock: 6, specialPrice: 130, prepTime: '20 mins', emoji: '🍚' },
  { id: 'V003', category: 'Veg', name: 'Aloo Gobi', description: 'Potato and cauliflower curry with Indian spices', price: 120, status: 'Available', isSpecial: false, stock: 10, specialPrice: null, prepTime: '12 mins', emoji: '🥔' },
  { id: 'V004', category: 'Veg', name: 'Chole Bhature', description: 'Spicy chickpeas served with fried bread', price: 140, status: 'Available', isSpecial: true, stock: 10, specialPrice: 120, prepTime: '15 mins', emoji: '🫓' },
  { id: 'V005', category: 'Veg', name: 'Palak Paneer', description: 'Cottage cheese cooked in creamy spinach sauce', price: 170, status: 'Available', isSpecial: false, stock: 18, specialPrice: null, prepTime: '15 mins', emoji: '🥬' },
  { id: 'V006', category: 'Veg', name: 'Masala Dosa', description: 'Crispy rice crepe stuffed with spiced potato filling', price: 110, status: 'Available', isSpecial: true, stock: 15, specialPrice: 90, prepTime: '10 mins', emoji: '🥞' },
  { id: 'V007', category: 'Veg', name: 'Veg Fried Rice', description: 'Fried rice with vegetables and soy sauce', price: 130, status: 'Available', isSpecial: false, stock: 13, specialPrice: null, prepTime: '10 mins', emoji: '🍳' },
  { id: 'V008', category: 'Veg', name: 'Veg Manchurian', description: 'Fried vegetable balls tossed in tangy sauce', price: 140, status: 'Available', isSpecial: false, stock: 8, specialPrice: null, prepTime: '12 mins', emoji: '🫙' },
  { id: 'V009', category: 'Veg', name: 'Paneer Tikka', description: 'Grilled marinated cottage cheese cubes', price: 160, status: 'Available', isSpecial: true, stock: 12, specialPrice: 140, prepTime: '10 mins', emoji: '🔥' },
  { id: 'V010', category: 'Veg', name: 'Mushroom Curry', description: 'Mushrooms cooked in rich gravy with spices', price: 150, status: 'Available', isSpecial: false, stock: 12, specialPrice: null, prepTime: '12 mins', emoji: '🍄' },
  // Non-Veg Items
  { id: 'NV001', category: 'Non-Veg', name: 'Chicken Biryani', description: 'Spiced rice with marinated chicken pieces', price: 200, status: 'Available', isSpecial: true, stock: 3, specialPrice: 180, prepTime: '20 mins', emoji: '🍗' },
  { id: 'NV002', category: 'Non-Veg', name: 'Mutton Rogan Josh', description: 'Tender mutton cooked in flavorful gravy', price: 280, status: 'Available', isSpecial: false, stock: 8, specialPrice: null, prepTime: '25 mins', emoji: '🥩' },
  { id: 'NV003', category: 'Non-Veg', name: 'Butter Chicken', description: 'Chicken cooked in buttery tomato-based sauce', price: 220, status: 'Available', isSpecial: true, stock: 8, specialPrice: 200, prepTime: '18 mins', emoji: '🍲' },
  { id: 'NV004', category: 'Non-Veg', name: 'Chicken Tikka', description: 'Marinated chicken grilled to perfection', price: 180, status: 'Available', isSpecial: false, stock: 11, specialPrice: null, prepTime: '12 mins', emoji: '🍢' },
  { id: 'NV005', category: 'Non-Veg', name: 'Fish Curry', description: 'Fresh fish cooked in spicy coastal gravy', price: 240, status: 'Available', isSpecial: false, stock: 5, specialPrice: null, prepTime: '20 mins', emoji: '🐟' },
  { id: 'NV006', category: 'Non-Veg', name: 'Prawn Masala', description: 'Juicy prawns in a rich spicy masala', price: 300, status: 'Available', isSpecial: false, stock: 4, specialPrice: null, prepTime: '20 mins', emoji: '🦐' },
  // Desserts
  { id: 'D001', category: 'Desserts', name: 'Gulab Jamun', description: 'Soft milk-solid dumplings soaked in sugar syrup', price: 80, status: 'Available', isSpecial: false, stock: 25, specialPrice: null, prepTime: '5 mins', emoji: '🟤' },
  { id: 'D002', category: 'Desserts', name: 'Rasmalai', description: 'Soft cottage cheese dumplings in creamy milk', price: 140, status: 'Available', isSpecial: true, stock: 10, specialPrice: 120, prepTime: '5 mins', emoji: '🍮' },
  { id: 'D003', category: 'Desserts', name: 'Kheer', description: 'Creamy rice pudding with nuts and cardamom', price: 90, status: 'Available', isSpecial: false, stock: 15, specialPrice: null, prepTime: '5 mins', emoji: '🥛' },
  // Beverages
  { id: 'BE001', category: 'Beverages', name: 'Masala Chai', description: 'Spiced Indian tea with milk', price: 40, status: 'Available', isSpecial: false, stock: 50, specialPrice: null, prepTime: '5 mins', emoji: '☕' },
  { id: 'BE002', category: 'Beverages', name: 'Fresh Lime Soda', description: 'Refreshing lime soda with mint', price: 60, status: 'Available', isSpecial: false, stock: 40, specialPrice: null, prepTime: '3 mins', emoji: '🍋' },
  { id: 'BE003', category: 'Beverages', name: 'Mango Lassi', description: 'Sweet mango yogurt drink', price: 100, status: 'Available', isSpecial: true, stock: 20, specialPrice: 80, prepTime: '5 mins', emoji: '🥭' },
  { id: 'BE004', category: 'Beverages', name: 'Cold Coffee', description: 'Chilled coffee blended with milk and ice cream', price: 120, status: 'Available', isSpecial: false, stock: 30, specialPrice: null, prepTime: '5 mins', emoji: '☕' },
  // Combo Meals
  { id: 'C001', category: 'Combos', name: 'Veg Thali', description: 'Complete vegetarian meal with dal, sabzi, roti, rice, dessert', price: 450, status: 'Available', isSpecial: true, stock: 8, specialPrice: 400, prepTime: '25 mins', emoji: '🍱' },
  { id: 'C002', category: 'Combos', name: 'Non-Veg Thali', description: 'Complete non-veg meal with chicken curry, roti, biryani', price: 550, status: 'Available', isSpecial: true, stock: 6, specialPrice: 500, prepTime: '30 mins', emoji: '🍛' },
  { id: 'C003', category: 'Combos', name: 'Family Pack (Veg)', description: 'Family meal for 4 with biryani, paneer, dal, bread', price: 1200, status: 'Available', isSpecial: true, stock: 3, specialPrice: 1000, prepTime: '35 mins', emoji: '👨‍👩‍👧‍👦' },
  { id: 'C004', category: 'Combos', name: 'Family Pack (Non-Veg)', description: 'Family feast with chicken biryani, mutton, naan, dessert', price: 1500, status: 'Available', isSpecial: true, stock: 2, specialPrice: 1200, prepTime: '40 mins', emoji: '🎉' },
];

export const TABLES = Array.from({ length: 20 }, (_, i) => ({
  id: `T${String(i + 1).padStart(2, '0')}`,
  status: 'Available',
  capacity: i < 5 ? 2 : i < 15 ? 4 : 6,
  section: i < 7 ? 'Ground Floor' : i < 14 ? 'First Floor' : 'Terrace',
}));

export const ANALYTICS = [
  { date: 'Jan 8', orders: 31, revenue: 18760, online: 19, cash: 12, avg: 605 },
  { date: 'Jan 9', orders: 27, revenue: 15940, online: 16, cash: 11, avg: 590 },
  { date: 'Jan 10', orders: 29, revenue: 17630, online: 18, cash: 11, avg: 608 },
  { date: 'Jan 11', orders: 32, revenue: 19420, online: 20, cash: 12, avg: 607 },
  { date: 'Jan 12', orders: 25, revenue: 14780, online: 15, cash: 10, avg: 591 },
  { date: 'Jan 13', orders: 35, revenue: 22150, online: 22, cash: 13, avg: 633 },
  { date: 'Jan 14', orders: 28, revenue: 16820, online: 16, cash: 12, avg: 601 },
  { date: 'Jan 15', orders: 30, revenue: 18450, online: 18, cash: 12, avg: 615 },
];

export const SPECIAL_ITEMS = [
  { id: 'SP001', name: 'Paneer Tikka', category: 'Appetizer', normalPrice: 280, discount: 14 },
  { id: 'SP002', name: 'Chicken Tikka', category: 'Appetizer', normalPrice: 320, discount: 13 },
  { id: 'SP007', name: 'Mango Lassi', category: 'Beverages', normalPrice: 100, discount: 20 },
  { id: 'SP009', name: 'Butter Chicken', category: 'Main Course', normalPrice: 380, discount: 16 },
  { id: 'SP010', name: 'Hara Bhara Kebab', category: 'Appetizer', normalPrice: 220, discount: 18 },
  { id: 'SP015', name: 'Family Pack (Non-Veg)', category: 'Combo', normalPrice: 1500, discount: 20 },
];

export const EMPLOYEES = [
  { id: 'EMP001', name: 'Rajesh Kumar', role: 'Restaurant Manager', dept: 'Management', shift: 'General', rating: 4.8, status: 'Active' },
  { id: 'EMP002', name: 'Priya Sharma', role: 'Head Chef', dept: 'Kitchen', shift: 'Morning', rating: 4.9, status: 'Active' },
  { id: 'EMP004', name: 'Anjali Patel', role: 'Cashier', dept: 'Front Desk', shift: 'Morning', rating: 4.5, status: 'Active' },
  { id: 'EMP005', name: 'Sanjay Reddy', role: 'Delivery Manager', dept: 'Delivery', shift: 'General', rating: 4.6, status: 'Active' },
];
